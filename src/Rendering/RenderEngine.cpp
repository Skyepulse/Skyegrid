#include "../../includes/Rendering/RenderEngine.hpp"
#include <iostream>
#include "../../includes/constants.hpp"

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
        voxelParams.voxelDimensions[0] = MAXIMUM_VOXEL_RESOLUTION;
        voxelParams.voxelDimensions[1] = MAXIMUM_VOXEL_RESOLUTION;
        voxelParams.voxelDimensions[2] = 1;
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