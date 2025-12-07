#include "../../includes/Rendering/RenderEngine.hpp"

//================================//
void RenderEngine::Render(void* userData)
{
    auto renderInfo = *static_cast<RenderInfo*>(userData);
    wgpu::SurfaceTexture currentTexture;
    this->wgpuBundle->GetSurface().GetCurrentTexture(&currentTexture);

    wgpu::RenderPassColorAttachment colorAttachment{};
    colorAttachment.view = currentTexture.texture.CreateView();
    colorAttachment.loadOp = wgpu::LoadOp::Clear;
    colorAttachment.clearValue = {0.3f, 0.3f, 0.3f, 1.0f};
    colorAttachment.storeOp = wgpu::StoreOp::Store;

    wgpu::RenderPassDescriptor renderPassDesc{};
    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &colorAttachment;

    wgpu::CommandEncoder encoder = this->wgpuBundle->GetDevice().CreateCommandEncoder();
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPassDesc);

    debugPipeline.AssertConsistent();
    pass.SetPipeline(debugPipeline.pipeline);
    pass.Draw(3); // Draw a triangle
    pass.End();

    wgpu::CommandBuffer commandBuffer = encoder.Finish();
    this->wgpuBundle->GetDevice().GetQueue().Submit(1, &commandBuffer);
}