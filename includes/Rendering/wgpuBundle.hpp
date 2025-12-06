#ifndef WGPU_BUNDLE_HPP
#define WGPU_BUNDLE_HPP

#include <webgpu/webgpu_cpp.h>
#include <dawn/webgpu_cpp_print.h>
#include <webgpu/webgpu_glfw.h>

//================================//
struct WindowFormat
{
    GLFWwindow* window;
    int width;
    int height;
};

//================================//
class WgpuBundle
{
public:
    WgpuBundle(WindowFormat windowFormat);
    ~WgpuBundle() = default;

    wgpu::Instance& GetInstance() { return this->instance; }
    wgpu::Adapter& GetAdapter() { return this->adapter; }
    wgpu::Device& GetDevice() { return this->device; }
    wgpu::Surface& GetSurface() { return this->surface; }
    wgpu::TextureFormat& GetSwapchainFormat() { return this->swapchainFormat; }

private:
    void InitializeInstance();
    void InitializeGraphics();

    void ConfigureSurface();

    // WebGPU objects
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;

    // Surface specifics
    wgpu::Surface surface;
    wgpu::TextureFormat swapchainFormat;

    // Window specifics
    GLFWwindow* window;
    int currentWidth;
    int currentHeight;
};

#endif // WGPU_BUNDLE_HPP