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
    std::vector<wgpu::BindGroupEntry> entries(7);

    entries[0].binding = 0;
    entries[0].textureView = this->computeVoxelPipeline.associatedTextureViews[0];

    entries[1].binding = 1;
    entries[1].buffer = this->computeVoxelPipeline.associatedUniforms[0];
    entries[1].offset = 0;
    entries[1].size = this->computeVoxelPipeline.uniformSizes[0];

    // Brick grid
    entries[2].binding = 2;
    entries[2].buffer = this->voxelManager->brickGridBuffer;
    entries[2].offset = 0;
    entries[2].size = this->voxelManager->brickGridBuffer.GetSize();

    // Brick pool
    entries[3].binding = 3;
    entries[3].buffer = this->voxelManager->brickPoolBuffer;
    entries[3].offset = 0;
    entries[3].size = this->voxelManager->brickPoolBuffer.GetSize();

    // Color pool
    entries[4].binding = 4;
    entries[4].buffer = this->voxelManager->colorPoolBuffer;
    entries[4].offset = 0;
    entries[4].size = this->voxelManager->colorPoolBuffer.GetSize();

    // Feedback buffer (Count)
    entries[5].binding = 5;
    entries[5].buffer = this->voxelManager->feedbackCountBuffer;
    entries[5].offset = 0;
    entries[5].size = this->voxelManager->feedbackCountBuffer.GetSize();

    // Feedback buffer (Indices)
    entries[6].binding = 6;
    entries[6].buffer = this->voxelManager->feedbackIndicesBuffer;
    entries[6].offset = 0;
    entries[6].size = this->voxelManager->feedbackIndicesBuffer.GetSize();

    wgpu::BindGroupDescriptor bindGroupDesc{};
    bindGroupDesc.layout = this->computeVoxelPipeline.bindGroupLayout;
    bindGroupDesc.entryCount = static_cast<uint32_t>(entries.size());
    bindGroupDesc.entries = entries.data();
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

    this->voxelManager->startOfFrame();
    const int num_voxels_max = MAXIMUM_VOXEL_RESOLUTION * MAXIMUM_VOXEL_RESOLUTION * MAXIMUM_VOXEL_RESOLUTION;
    if (this->voxelManager->lastBrickIndex >= num_voxels_max)
        this->voxelManager->lastBrickIndex = 0;

    if (this->voxelManager->lastBrickIndex < num_voxels_max)
    {
        int x = (this->voxelManager->lastBrickIndex % MAXIMUM_VOXEL_RESOLUTION);;
        int y = (this->voxelManager->lastBrickIndex / MAXIMUM_VOXEL_RESOLUTION) % MAXIMUM_VOXEL_RESOLUTION;;
        int z = (this->voxelManager->lastBrickIndex / (MAXIMUM_VOXEL_RESOLUTION * MAXIMUM_VOXEL_RESOLUTION)) % MAXIMUM_VOXEL_RESOLUTION;
        // Color rgb, from red to green depending on index pure red at 0,0,0 and pure green at max,max,max
        ColorRGB color = {
            uint8_t(255 - (this->voxelManager->lastBrickIndex * 255) / num_voxels_max),
            uint8_t((this->voxelManager->lastBrickIndex * 255) / num_voxels_max),
            0,
            0
        };
        this->voxelManager->setVoxel(x, y, z, true, color);
    }

    this->voxelManager->lastBrickIndex += 1;

    wgpu::TextureView swapchainView = currentTexture.texture.CreateView();

    // Command Encoder
    wgpu::Queue queue = this->wgpuBundle->GetDevice().GetQueue();
    wgpu::CommandEncoderDescriptor cmdDesc{};
    wgpu::CommandEncoder encoder = this->wgpuBundle->GetDevice().CreateCommandEncoder(&cmdDesc);

    // Update voxel data:
    this->voxelManager->update(*this->wgpuBundle, queue, encoder);

    // Upload pass
    const uint32_t uploadCount = this->voxelManager->pendingUploadCount;
    this->computeUploadVoxelPipeline.AssertConsistent();
    {
        // Write uniform
        queue.WriteBuffer(
            this->voxelManager->uploadCountUniform,
            0,
            &uploadCount,
            sizeof(uint32_t)
        );

        wgpu::ComputePassDescriptor computePassDesc{};
        computePassDesc.timestampWrites = nullptr;
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass(&computePassDesc);

        this->computeUploadVoxelPipeline.AssertInitialized();
        pass.SetPipeline(this->computeUploadVoxelPipeline.computePipeline);
        pass.SetBindGroup(0, this->computeUploadVoxelPipeline.bindGroup);

        // Dispatch size based on upload buffer size, filled by CPU
        if (uploadCount > 0)
        {
            uint32_t dispatchX = (uploadCount + 127) / 128;
            pass.DispatchWorkgroups(dispatchX, 1, 1);
        }

        pass.End();
    }

    // Compute pass (Raytracing)
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

    this->voxelManager->prepareFeedback(queue, encoder);

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

    // Wait for all futures
    wgpu::Future readFeedbackFuture;
    this->voxelManager->readFeedback(readFeedbackFuture);

    wgpu::Future remapUploadFuture;
    bool remapRequested = this->voxelManager->remapUploadBuffer(remapUploadFuture);

    this->wgpuBundle->GetInstance().WaitAny(readFeedbackFuture, UINT64_MAX);
    if (remapRequested)
        this->wgpuBundle->GetInstance().WaitAny(remapUploadFuture, UINT64_MAX);
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