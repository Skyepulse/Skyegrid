#include "../../includes/Rendering/RenderEngine.hpp"
#include <iostream>
#include "../../includes/constants.hpp"
#include <time.h>
#include <numeric>

//================================//
void RenderEngine::InitImGui()
{
    std::cout << "[RenderEngine] Initializing ImGui...\n";
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    GLFWwindow* window = this->wgpuBundle->GetGLFWWindow();

    ImGui_ImplGlfw_InitForOther(window, true);

    wgpu::Device device = this->wgpuBundle->GetDevice();
    WindowFormat windowFormat = this->wgpuBundle->GetWindowFormat();

    ImGui_ImplWGPU_InitInfo init_info = {};
    init_info.Device = device.Get();
    init_info.NumFramesInFlight = 3;
    init_info.RenderTargetFormat = WGPUTextureFormat_BGRA8Unorm;

    ImGui_ImplWGPU_Init(&init_info);
    std::cout << "[RenderEngine] ImGui initialized successfully.\n";
}

//================================//
void RenderEngine::RenderImGui(wgpu::RenderPassEncoder& pass)
{
    ImGui_ImplWGPU_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Voxel Controls");

    ImGui::Text("Voxel Resolution: %d", this->GetVoxelResolution());
    ImGui::Separator();

    ImGui::SliderInt("Voxel resolution", &resolutionSliderValue, 1, 2048);
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        if (resolutionSliderValue != previousResolutionSliderValue)
        {
            // Change only resolution
            this->voxelManager->ChangeVoxelResolution(*this->wgpuBundle, resolutionSliderValue);
            resolutionSliderValue = this->voxelManager->GetVoxelResolution();
            visibleBricksSliderValue = this->voxelManager->GetMaxVisibleBricks();

            // recreate bind groups (needed)
            this->resizePending = true;
            this->voxelManager->createUploadBindGroup(this->computeUploadVoxelPipeline, *this->wgpuBundle);

            previousResolutionSliderValue = resolutionSliderValue;
        }
    }

    ImGui::Separator();
    ImGui::Text("Max Visible Bricks: %d", this->voxelManager->GetMaxVisibleBricks());
    ImGui::Separator();

    ImGui::SliderInt("Max Visible Bricks", &visibleBricksSliderValue, 1, 3000000);
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        if (visibleBricksSliderValue != previousVisibleBricksSliderValue)
        {
            // Change only max visible bricks
            this->voxelManager->ChangeVoxelResolution(*this->wgpuBundle, this->GetVoxelResolution(), visibleBricksSliderValue);
            visibleBricksSliderValue = this->voxelManager->GetMaxVisibleBricks();
            resolutionSliderValue = this->voxelManager->GetVoxelResolution();

            // recreate bind groups (needed)
            this->resizePending = true;
            this->voxelManager->createUploadBindGroup(this->computeUploadVoxelPipeline, *this->wgpuBundle);

            previousVisibleBricksSliderValue = visibleBricksSliderValue;
        }
    }

    ImGui::Separator();
    ImGui::Text("CPU Frame Time: %.3f ms", this->cpuFrameTimeMS);
    ImGui::Separator();
    ImGui::Text("GPU Ray Trace Time: %.3f ms", this->gpuFrameTimeRayTraceMs);
    ImGui::Text("GPU Upload Time: %.3f ms", this->gpuFrameTimeUploadMs);
    ImGui::Text("GPU Blit Time: %.3f ms", this->gpuFrameTimeBlitMs);
    ImGui::Separator();

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();

    ImGui::Render();
    ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass.Get());
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
    std::vector<wgpu::BindGroupEntry> entries(9);

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

    // Feedback buffer (Count)
    entries[4].binding = 4;
    entries[4].buffer = this->voxelManager->feedbackCountBuffer;
    entries[4].offset = 0;
    entries[4].size = this->voxelManager->feedbackCountBuffer.GetSize();

    // Feedback buffer (Indices)
    entries[5].binding = 5;
    entries[5].buffer = this->voxelManager->feedbackIndicesBuffer;
    entries[5].offset = 0;
    entries[5].size = this->voxelManager->feedbackIndicesBuffer.GetSize();

    // Color pool
    for (int i = 0; i < MAX_COLOR_POOLS; ++i)
    {
        entries[6 + i].binding = 6 + i;
        entries[6 + i].buffer = this->voxelManager->colorPoolBuffers[i];
        entries[6 + i].offset = 0;
        entries[6 + i].size = this->voxelManager->colorPoolBuffers[i].GetSize();
    }

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

    auto cpuFrameStart = std::chrono::high_resolution_clock::now();

    // PROCESS ASYNC OPERATIONS
    wgpu::Instance instance = this->wgpuBundle->GetInstance();
    this->voxelManager->processAsyncOperations(instance);
    this->voxelManager->startOfFrame();

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
        UploadUniform uploadUniform{};
        uploadUniform.uploadCount = uploadCount;
        uploadUniform.maxColorBufferSize = this->voxelManager->maxColorBufferEntries;
        uploadUniform.hasColor = this->voxelManager->GetHasColor() ? 1 : 0;

        // Write uniform
        queue.WriteBuffer(
            this->voxelManager->uploadCountUniform,
            0,
            &uploadUniform,
            sizeof(UploadUniform)
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
        voxelParams.maxColorBufferSize = this->voxelManager->maxColorBufferEntries;
        voxelParams.voxelResolution = static_cast<uint32_t>(this->voxelManager->GetVoxelResolution());
        voxelParams.time = static_cast<float>(renderInfo.time);
        voxelParams.hasColor = this->voxelManager->GetHasColor() ? 1 : 0;

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

        // ImGui rendering
        this->RenderImGui(pass);

        pass.End();
    }

    wgpu::CommandBuffer commandBuffer = encoder.Finish();
    this->wgpuBundle->GetDevice().GetQueue().Submit(1, &commandBuffer);

    // After submit, request mapping of the feedback buffer we just wrote to
    // APPARENTLY THIS WILL NOT BLOCK! The callback will fire when the GPU is DONE
    int feedbackSlot = this->voxelManager->currentFeedbackReadSlot;
    if (this->voxelManager->feedbackBufferSlots[feedbackSlot].state == BufferState::Available)
    {
        this->voxelManager->feedbackBufferSlots[feedbackSlot].state = BufferState::MappingInFlight;
        
        // Create context for callback
        struct FeedbackCtx {
            VoxelManager* vm;
            int slot;
        };
        auto* ctx = new FeedbackCtx{this->voxelManager.get(), feedbackSlot};
        
        size_t bufferSize = sizeof(uint32_t) + MAX_FEEDBACK * sizeof(uint32_t);
        
        this->voxelManager->feedbackBufferSlots[feedbackSlot].cpuBuffer.MapAsync(
            wgpu::MapMode::Read,
            0,
            bufferSize,
            wgpu::CallbackMode::AllowProcessEvents,
            [](wgpu::MapAsyncStatus status, wgpu::StringView message, FeedbackCtx* ctx) {
                FeedbackBufferSlot& slot = ctx->vm->feedbackBufferSlots[ctx->slot];
                
                if (status == wgpu::MapAsyncStatus::Success)
                {
                    const uint8_t* mappedData = static_cast<const uint8_t*>(
                        slot.cpuBuffer.GetConstMappedRange()
                    );
                    
                    uint32_t count = *reinterpret_cast<const uint32_t*>(mappedData);
                    count = std::min(count, static_cast<uint32_t>(MAX_FEEDBACK));
                    
                    const uint32_t* indices = reinterpret_cast<const uint32_t*>(
                        mappedData + sizeof(uint32_t)
                    );
                    
                    ctx->vm->feedbackRequests.resize(count);
                    if (count > 0)
                    {
                        memcpy(ctx->vm->feedbackRequests.data(), indices, count * sizeof(uint32_t));
                    }
                    
                    ctx->vm->hasPendingFeedback = true;
                    
                    slot.cpuBuffer.Unmap();
                }
                
                slot.state = BufferState::Available;
                delete ctx;
            },
            ctx
        );
    }

    auto cpuFrameEnd = std::chrono::high_resolution_clock::now();
    double cpuFrameMs = std::chrono::duration<double, std::milli>(cpuFrameEnd - cpuFrameStart).count();
    this->cpuFrameAccumulator.push_back(static_cast<float>(cpuFrameMs));
    if (this->cpuFrameAccumulator.size() > 10)
        this->cpuFrameAccumulator.erase(this->cpuFrameAccumulator.begin());
    this->cpuFrameTimeMS = std::accumulate(this->cpuFrameAccumulator.begin(), this->cpuFrameAccumulator.end(), 0.0f) / static_cast<float>(this->cpuFrameAccumulator.size());
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