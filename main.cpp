#include <iostream>
#include "includes/Rendering/wgpuBundle.hpp"
#include "includes/Rendering/Pipelines/pipelines.hpp"
#include <GLFW/glfw3.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

const int kWidth = 512;
const int kHeight = 512;

RenderPipelineWrapper debugPipeline;

//================================//
void Render(void* userData)
{
    auto& wgpuBundle = *static_cast<WgpuBundle*>(userData);
    wgpu::SurfaceTexture currentTexture;
    wgpuBundle.GetSurface().GetCurrentTexture(&currentTexture);

    wgpu::RenderPassColorAttachment colorAttachment{};
    colorAttachment.view = currentTexture.texture.CreateView();
    colorAttachment.loadOp = wgpu::LoadOp::Clear;
    colorAttachment.clearValue = {0.3f, 0.3f, 0.3f, 1.0f};
    colorAttachment.storeOp = wgpu::StoreOp::Store;

    wgpu::RenderPassDescriptor renderPassDesc{};
    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &colorAttachment;

    wgpu::CommandEncoder encoder = wgpuBundle.GetDevice().CreateCommandEncoder();
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPassDesc);

    pass.SetPipeline(debugPipeline.pipeline);
    pass.Draw(3); // Draw a triangle
    pass.End();

    wgpu::CommandBuffer commandBuffer = encoder.Finish();
    wgpuBundle.GetDevice().GetQueue().Submit(1, &commandBuffer);
}

//================================//
int main()
{
    std::cout << "[MAIN] Hello, Skyegrid!" << std::endl;

    if (!glfwInit())
    {
        std::cerr << "[MAIN] Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> window(
        glfwCreateWindow(kWidth, kHeight, "Skyegrid", nullptr, nullptr),
        &glfwDestroyWindow
    );

    if (!window)
    {
        std::cerr << "[MAIN] Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    try
    {
        WindowFormat windowFormat = { window.get(), kWidth, kHeight };
        WgpuBundle wgpuBundle(windowFormat);
        std::cout << "[MAIN] WebGPU initialized successfully." << std::endl;

        CreateRenderPipelineDebug(wgpuBundle, debugPipeline);

#if defined(__EMSCRIPTEN__)
        emscripten_set_main_loop_arg(Render, &wgpuBundle, 0, true);
#else
        while (!glfwWindowShouldClose(window.get()))
        {
            glfwPollEvents();
            Render(&wgpuBundle);
            wgpuBundle.GetSurface().Present(); // Swapchain present
            wgpuBundle.GetInstance().ProcessEvents();
        }
#endif
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << "[MAIN] " << e.what() << std::endl;
        glfwDestroyWindow(window.get());
        glfwTerminate();
        return -1;
    }

    glfwDestroyWindow(window.get());
    glfwTerminate();
    return 0;
}