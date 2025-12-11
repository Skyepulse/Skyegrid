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
void RenderEngine::RebuildVoxelPipelineResources(const RenderInfo& renderInfo)
{
    // We reconstruct here the objects
    // of the voxel compute pipeline that depend on window size directly
    // Recreate output texture for voxel pipeline
    this->computeVoxelPipeline.textureSizes[0] = renderInfo.width * renderInfo.height * 4; // Output voxel texture, RGBA8
    this->computeVoxelPipeline.associatedTextures[0] = nullptr;
    this->computeVoxelPipeline.associatedTextureViews[0] = nullptr;

    wgpu::TextureDescriptor textureDescriptor{};
    wgpu::TextureViewDescriptor viewDescriptor{};

    textureDescriptor.dimension = wgpu::TextureDimension::e2D;
    textureDescriptor.size = { renderInfo.width, renderInfo.height, 1 };
    textureDescriptor.sampleCount = 1;
    textureDescriptor.mipLevelCount = 1;
    textureDescriptor.format = wgpu::TextureFormat::RGBA8Unorm;
    textureDescriptor.usage = wgpu::TextureUsage::StorageBinding | wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::TextureBinding;
    this->computeVoxelPipeline.associatedTextures[0] = this->wgpuBundle->GetDevice().CreateTexture(&textureDescriptor);

    viewDescriptor.dimension = wgpu::TextureViewDimension::e2D;
    viewDescriptor.format = textureDescriptor.format;
    this->computeVoxelPipeline.associatedTextureViews[0] = this->computeVoxelPipeline.associatedTextures[0].CreateView(&viewDescriptor);

    // Compute pipeline bind group
    // Bind Group
    this->computeVoxelPipeline.bindGroup = nullptr;
    std::vector<wgpu::BindGroupEntry> bindGroupEntries(3);

    bindGroupEntries[0].binding = 0;
    bindGroupEntries[0].textureView = this->computeVoxelPipeline.associatedTextureViews[0];

    bindGroupEntries[1].binding = 1;
    bindGroupEntries[1].textureView = this->computeVoxelPipeline.associatedTextureViews[1];

    bindGroupEntries[2].binding = 2;
    bindGroupEntries[2].buffer = this->computeVoxelPipeline.associatedUniforms[0];
    bindGroupEntries[2].offset = 0;
    bindGroupEntries[2].size = this->computeVoxelPipeline.uniformSizes[0];

    wgpu::BindGroupDescriptor bindGroupDesc{};
    bindGroupDesc.layout = this->computeVoxelPipeline.bindGroupLayout;
    bindGroupDesc.entryCount = static_cast<uint32_t>(bindGroupEntries.size());
    bindGroupDesc.entries = bindGroupEntries.data();
    this->computeVoxelPipeline.bindGroup = this->wgpuBundle->GetDevice().CreateBindGroup(&bindGroupDesc);

    // Blit pipeline bind group
    this->blitVoxelPipeline.bindGroup = nullptr; 
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
}

//================================//
void RenderEngine::PackVoxelDataToGPU()
{
    std::cout << "Packing voxel data to GPU...\n";

    std::vector<uint32_t> packed;
    PackVoxelsRGBA32UI(this->voxelDataCache, MAXIMUM_VOXEL_RESOLUTION, packed);

    std::cout << "Total size of packed voxel data in bytes: " << packed.size() * sizeof(uint32_t) << "\n";

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

    this->wgpuBundle->GetDevice().GetQueue().WriteTexture(
        &textureCopyDesc,
        packed.data(),
        packed.size() * sizeof(uint32_t),
        &bufferLayout,
        &copySize
    );

    std::cout << "Voxel data packed to GPU successfully.\n";
}

//================================//
void RenderEngine::Render(void* userData)
{
    auto renderInfo = *static_cast<RenderInfo*>(userData);
    if (renderInfo.resizeNeeded)
        this->resizePending = true;

    if (this->resizePending) // resizePending is true on start, so on first frame call
    {
        this->resizePending = false;
        
        // We will have to recreate the output texture and bind group here
        this->RebuildVoxelPipelineResources(renderInfo);

        // Camera resize
        WindowFormat windowFormat = this->wgpuBundle->GetWindowFormat();
        this->camera->SetExtent(Eigen::Vector2f(static_cast<float>(windowFormat.width), static_cast<float>(windowFormat.height)));
    }

    // Swapchain Texture View    
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
    this->computeVoxelPipeline.AssertConsistent();
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
    this->blitVoxelPipeline.AssertConsistent();
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

//================================//
void RenderEngine::RenderDebug(void* userData)
{
    auto renderInfo = *static_cast<RenderInfo*>(userData);
    if (renderInfo.resizeNeeded)
        this->resizePending = true;

    if (this->resizePending) // resizePending is true on start, so on first frame call
    {
        this->resizePending = false;
        
        // Camera resize
        WindowFormat windowFormat = this->wgpuBundle->GetWindowFormat();
        this->camera->SetExtent(Eigen::Vector2f(static_cast<float>(windowFormat.width), static_cast<float>(windowFormat.height)));
    }

    // Swapchain Texture View    
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

    // Debug render pass
    this->debugPipeline.AssertConsistent();
    {
        wgpu::RenderPassColorAttachment colorAttachment{};
        colorAttachment.view = swapchainView;
        colorAttachment.loadOp  = wgpu::LoadOp::Clear;
        colorAttachment.clearValue = { 0.1f, 0.1f, 0.1f, 1.0f };
        colorAttachment.storeOp = wgpu::StoreOp::Store;

        wgpu::RenderPassDescriptor renderPassDesc{};
        renderPassDesc.colorAttachmentCount = 1;
        renderPassDesc.colorAttachments     = &colorAttachment;

        wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPassDesc);

        this->debugPipeline.AssertInitialized();
        pass.SetPipeline(this->debugPipeline.pipeline);

        // Fullscreen triangle
        pass.Draw(3);
        pass.End();
    }
    wgpu::CommandBuffer commandBuffer = encoder.Finish();
    this->wgpuBundle->GetDevice().GetQueue().Submit(1, &commandBuffer);
}