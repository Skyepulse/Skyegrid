#include "../../includes/Rendering/RenderEngine.hpp"
#include <iostream>
#include "../../includes/constants.hpp"

//================================//
static void PackVoxelsRGBA32UI(
    const std::vector<std::vector<std::vector<uint8_t>>>& voxels,
    uint32_t resolution,
    std::vector<uint32_t>& outTexels
)
{
    const uint32_t texX = resolution / 4;
    const uint32_t texY = resolution / 4;
    const uint32_t texZ = resolution / 8;

    outTexels.assign(texX * texY * texZ * 4, 0u);

    auto texelIndex = [&](uint32_t x, uint32_t y, uint32_t z) {
        return (x + y * texX + z * texX * texY) * 4;
    };

    for (uint32_t z = 0; z < resolution; ++z)
    for (uint32_t y = 0; y < resolution; ++y)
    for (uint32_t x = 0; x < resolution; ++x)
    {
        if (!voxels[x][y][z])
            continue;

        uint32_t tx = x / 4;
        uint32_t ty = y / 4;
        uint32_t tz = z / 8;

        uint32_t channel = x % 4;
        uint32_t bit =
            (y % 4) +
            (z % 8) * 4;

        outTexels[texelIndex(tx, ty, tz) + channel] |= (1u << bit);
    }
}

//================================//
void RenderEngine::Render(void* userData)
{
    if (this->resizePending) // resizePending is true on start, so on first frame call
    {
        // We will have to recreate the output texture and bind group here
        this->resizePending = false;

        // For now we wont resize anything, we skip the recreation. We suppose the 
        // texture has been correctly created at pipeline creation for now.

        this->blitVoxelPipeline.bindGroup = nullptr; // Reset bind group to recreate it
        wgpu::BindGroupEntry blitEntries[2]{};
        blitEntries[0].binding = 0;
        blitEntries[0].textureView = this->computeVoxelPipeline.associatedTextureViews[0];
        blitEntries[1].binding = 1;
        blitEntries[1].sampler = this->blitVoxelPipeline.associatedSamplers[0];
        wgpu::BindGroupDescriptor blitBindGroupDesc{};
        blitBindGroupDesc.layout = this->blitVoxelPipeline.bindGroupLayout;
        blitBindGroupDesc.entryCount = 2;
        blitBindGroupDesc.entries = blitEntries;
        this->blitVoxelPipeline.bindGroup = this->wgpuBundle->GetDevice().CreateBindGroup(&blitBindGroupDesc);

        // Camera resize
        WindowFormat windowFormat = this->wgpuBundle->GetWindowFormat();
        this->camera->SetExtent(Eigen::Vector2f(static_cast<float>(windowFormat.width), static_cast<float>(windowFormat.height)));
    }

    // Swapchain Texture View
    auto renderInfo = *static_cast<RenderInfo*>(userData);
    
    wgpu::SurfaceTexture currentTexture;
    wgpu::Surface surface = this->wgpuBundle->GetSurface();
    surface.GetCurrentTexture(&currentTexture);

    auto status = currentTexture.status;
    if (status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal &&
        status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal)
    {
        std::cout << "Surface lost/outdated, need reconfigure.\n";
        return;
    }

    wgpu::TextureView swapchainView = currentTexture.texture.CreateView();

    // Command Encoder
    wgpu::Queue queue = this->wgpuBundle->GetDevice().GetQueue();
    wgpu::CommandEncoderDescriptor cmdDesc{};
    wgpu::CommandEncoder encoder = this->wgpuBundle->GetDevice().CreateCommandEncoder(&cmdDesc);

    // Compute pass
    {
        // Write Uniform
        VoxelParameters voxelParams{};
        voxelParams.pixelToRay = this->camera->PixelToRayMatrix();
        voxelParams.cameraOrigin = this->camera->GetPosition();
        voxelParams.voxelResolution = MAXIMUM_VOXEL_RESOLUTION;
        voxelParams.time = static_cast<float>(renderInfo.time);

        queue.WriteBuffer(
            this->computeVoxelPipeline.associatedUniforms[0],
            0,
            &voxelParams,
            sizeof(VoxelParameters)
        );

        // Write Voxel Data into this->computeVoxelPipeline.associatedTextures[0] and associatedTextureViews[0]
        std::vector<uint32_t> packed;
        PackVoxelsRGBA32UI(this->voxelDataCache, MAXIMUM_VOXEL_RESOLUTION, packed);

        wgpu::TexelCopyTextureInfo textureCopyDesc{};
        textureCopyDesc.texture = this->computeVoxelPipeline.associatedTextures[1];
        textureCopyDesc.mipLevel = 0;
        textureCopyDesc.origin = { 0, 0, 0 };
        textureCopyDesc.aspect = wgpu::TextureAspect::All;

        wgpu::TexelCopyBufferLayout bufferLayout{};
        bufferLayout.offset = 0;
        bufferLayout.bytesPerRow = MAXIMUM_VOXEL_RESOLUTION / 4 * 4 * sizeof(uint32_t);
        bufferLayout.rowsPerImage = MAXIMUM_VOXEL_RESOLUTION / 4;

        wgpu::Extent3D copySize{};
        copySize.width = MAXIMUM_VOXEL_RESOLUTION / 4;
        copySize.height = MAXIMUM_VOXEL_RESOLUTION / 4;
        copySize.depthOrArrayLayers = MAXIMUM_VOXEL_RESOLUTION / 8;

        queue.WriteTexture(
            &textureCopyDesc,
            packed.data(),
            packed.size() * sizeof(uint32_t),
            &bufferLayout,
            &copySize
        );

        wgpu::ComputePassDescriptor computePassDesc{};
        computePassDesc.timestampWrites = nullptr;
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass(&computePassDesc);

        this->computeVoxelPipeline.AssertInitialized();
        pass.SetPipeline(this->computeVoxelPipeline.computePipeline);
        pass.SetBindGroup(0, this->computeVoxelPipeline.bindGroup);

        uint32_t dispatchX = (renderInfo.width  + 7) / 8;
        uint32_t dispatchY = (renderInfo.height  + 7) / 8;

        pass.DispatchWorkgroups(dispatchX, dispatchY, 1);
        pass.End();
    }

    // Blit pass
    {
        wgpu::RenderPassColorAttachment colorAttachment{};
        colorAttachment.view = swapchainView;
        colorAttachment.loadOp  = wgpu::LoadOp::Clear;
        colorAttachment.clearValue = { 0.0f, 0.0f, 0.0f, 1.0f };
        colorAttachment.storeOp = wgpu::StoreOp::Store;

        wgpu::RenderPassDescriptor renderPassDesc{};
        renderPassDesc.colorAttachmentCount = 1;
        renderPassDesc.colorAttachments     = &colorAttachment;

        wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPassDesc);

        this->blitVoxelPipeline.AssertInitialized();
        pass.SetPipeline(this->blitVoxelPipeline.pipeline);
        pass.SetBindGroup(0, this->blitVoxelPipeline.bindGroup);

        // Fullscreen triangle
        pass.Draw(3);

        pass.End();
    }

    wgpu::CommandBuffer commandBuffer = encoder.Finish();
    this->wgpuBundle->GetDevice().GetQueue().Submit(1, &commandBuffer);
}