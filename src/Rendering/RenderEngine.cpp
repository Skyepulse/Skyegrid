#include "../../includes/Rendering/RenderEngine.hpp"
#include <iostream>
#include "../../includes/constants.hpp"
#include <time.h>

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
void RenderEngine::SetPackedVoxel(uint32_t x, uint32_t y, uint32_t z, bool on)
{
    const uint32_t res = MAXIMUM_VOXEL_RESOLUTION;
    const uint32_t texX = res / 4;
    const uint32_t texY = res / 4;

    const uint32_t desiredX = x / 4;
    const uint32_t desiredY = y / 4;
    const uint32_t desiredZ = z / 8;

    const uint32_t channel = x % 4;
    const uint32_t bit = (y % 4) + (z % 8) * 4; // 0..31
    const uint32_t mask = 1u << bit;

    const uint32_t texelBase = (desiredX + desiredY * texX + desiredZ * texX * texY) * 4;

    // Get reference to the correct word
    uint32_t& word = this->texelInfo[texelBase + channel];

    if (on) word |= mask;
    else    word &= ~mask;
}

//================================//
void RenderEngine::PackVoxelDataToGPU()
{
    std::cout << "Packing voxel data to GPU...\n";
    const float startTime = static_cast<float>(clock()) / CLOCKS_PER_SEC;

    const size_t res = static_cast<size_t>(MAXIMUM_VOXEL_RESOLUTION);
    const uint32_t texX = res / 4;
    const uint32_t texY = res / 4;
    const uint32_t texZ = res / 8;

    texelInfo.assign(texX * texY * texZ * 4, 0u);

    // Cube edges
    for (int x = 0; x < MAXIMUM_VOXEL_RESOLUTION; ++x)
    for (int y = 0; y < MAXIMUM_VOXEL_RESOLUTION; ++y)
    for (int z = 0; z < MAXIMUM_VOXEL_RESOLUTION; ++z)
    {
        bool edge =
            // edges parallel to X
            ((y == 0 || y == MAXIMUM_VOXEL_RESOLUTION - 1) &&
            (z == 0 || z == MAXIMUM_VOXEL_RESOLUTION - 1)) ||

            // edges parallel to Y
            ((x == 0 || x == MAXIMUM_VOXEL_RESOLUTION - 1) &&
            (z == 0 || z == MAXIMUM_VOXEL_RESOLUTION - 1)) ||

            // edges parallel to Z
            ((x == 0 || x == MAXIMUM_VOXEL_RESOLUTION - 1) &&
            (y == 0 || y == MAXIMUM_VOXEL_RESOLUTION - 1));

        if (edge)
            this->SetPackedVoxel(x, y, z, true);
    }

    // Cube in the exact middle
    for (int x = MAXIMUM_VOXEL_RESOLUTION / 4; x < 3 * MAXIMUM_VOXEL_RESOLUTION / 4; ++x)
    for (int y = MAXIMUM_VOXEL_RESOLUTION / 4; y < 3 * MAXIMUM_VOXEL_RESOLUTION / 4; ++y)
    for (int z = MAXIMUM_VOXEL_RESOLUTION / 4; z < 3 * MAXIMUM_VOXEL_RESOLUTION / 4; ++z)
    {
        this->SetPackedVoxel(x, y, z, true);
    }
    
    std::cout << "Total size of packed voxel data in bytes: " << texelInfo.size() * sizeof(uint32_t) << "\n";

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
        this->texelInfo.data(),
        this->texelInfo.size() * sizeof(uint32_t),
        &bufferLayout,
        &copySize
    );

    const float endTime = static_cast<float>(clock()) / CLOCKS_PER_SEC;
    std::cout << "Voxel data packing took " << (endTime - startTime) << " seconds.\n";
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