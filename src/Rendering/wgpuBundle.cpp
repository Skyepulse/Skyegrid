#include "../../includes/Rendering/wgpuBundle.hpp"
#include "../../includes/Rendering/wgpuHelpers.hpp"
#include "../../includes/constants.hpp"

#include <GLFW/glfw3.h>
#include <vector>
#include <iostream>

//================================//
WgpuBundle::WgpuBundle(WindowFormat windowFormat) : window(windowFormat.window), currentWidth(windowFormat.width), currentHeight(windowFormat.height)
{
    this->InitializeInstance();
    this->surface = wgpu::glfw::CreateSurfaceForWindow(this->instance, window);

    // Callback on window resize
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(
        window,
        [](GLFWwindow* wnd, int width, int height)
        {
            WgpuBundle* bundle = static_cast<WgpuBundle*>(glfwGetWindowUserPointer(wnd));
            bundle->Resize(width, height);
        }
    );

    this->InitializeGraphics();
}

//================================//
void WgpuBundle::InitializeInstance()
{
    // Instance required features
    const wgpu::InstanceFeatureName kTimeWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
    std::vector<wgpu::InstanceFeatureName> requiredFeatures = { kTimeWaitAny };

    if (wgpuCreateInstance(this->instance, requiredFeatures) < 0)
        throw std::runtime_error("Failed to create WebGPU instance.");

    // Adapter
    wgpu::RequestAdapterOptions options{};
    options.backendType = wgpu::BackendType::Null;

    if (wgpuRequestAdapter(this->instance, this->adapter, &options) < 0)
        throw std::runtime_error("Failed to request WebGPU adapter.");

    adapter.GetLimits(&this->limits);
    this->ComputeLimits();

    // Device
    wgpu::DeviceDescriptor deviceDesc{};
    deviceDesc.requiredLimits = &this->limits;
    deviceDesc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType errorType, wgpu::StringView message)
        {
            std::cout << "[wgpuDevice] Uncaptured error: " << message << std::endl;
        }
    );
    if (wgpuCreateDevice(this->instance, this->adapter, this->device, &deviceDesc) < 0)
        throw std::runtime_error("Failed to create WebGPU device.");

    wgpu::AdapterInfo info;
    adapter.GetInfo(&info);
    std::cout << "[wgpuBundle][Init] Using adapter: " << info.description << std::endl;
    std::cout << "[wgpuBundle][Init] Using device: " << info.device << std::endl;
    std::cout << "[wgpuBundle][Init] Backend: " << static_cast<uint32_t>(info.backendType) << std::endl;
}

//================================//
void WgpuBundle::InitializeGraphics()
{
    this->ConfigureSurface();
}

//================================//
void WgpuBundle::ConfigureSurface()
{
    wgpu::SurfaceCapabilities capabilities;
    surface.GetCapabilities(adapter, &capabilities);
    swapchainFormat = capabilities.formats[0];

    wgpu::SurfaceConfiguration config{};
    config.device = device;
    config.format = swapchainFormat;
    config.width = currentWidth;
    config.height = currentHeight;
    config.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc;

    surface.Configure(&config);
}

//================================//
void WgpuBundle::ComputeLimits()
{
    // Adapter limits MUST already be loaded via:
    // adapter.GetLimits(&this->limits);

    // ---------------------------------------
    // DO NOT TOUCH BUFFER LIMITS (CRITICAL)
    // ---------------------------------------
    // Dawn uses large internal storage buffers.
    // Reducing these causes the 2GB uploader crash.

    // this->limits.maxBufferSize               <-- leave untouched
    // this->limits.maxStorageBufferBindingSize <-- leave untouched
    // this->limits.maxStorageBuffersPerShaderStage <-- leave untouched


    // ---------------------------------------
    // STORAGE TEXTURES (you use exactly 2)
    // ---------------------------------------
    this->limits.maxStorageTexturesPerShaderStage =
        std::max(this->limits.maxStorageTexturesPerShaderStage, 2u);


    // ---------------------------------------
    // TEXTURE DIMENSIONS
    // ---------------------------------------
    // These must be >= what you create

    this->limits.maxTextureDimension2D =
        std::max(this->limits.maxTextureDimension2D,
                 static_cast<uint32_t>(
                     std::max(MAXIMUM_WINDOW_WIDTH, MAXIMUM_WINDOW_HEIGHT)));

    this->limits.maxTextureDimension3D =
        std::max(this->limits.maxTextureDimension3D,
                 static_cast<uint32_t>(MAXIMUM_VOXEL_RESOLUTION / 4));


    // ---------------------------------------
    // UNIFORMS (small, aligned)
    // ---------------------------------------
    this->limits.maxUniformBuffersPerShaderStage =
        std::max(this->limits.maxUniformBuffersPerShaderStage, 1u);

    this->limits.maxUniformBufferBindingSize =
        std::max(static_cast<uint32_t>(this->limits.maxUniformBufferBindingSize), 256u);


    // ---------------------------------------
    // COMPUTE LIMITS
    // ---------------------------------------
    this->limits.maxComputeWorkgroupSizeX =
        std::max(this->limits.maxComputeWorkgroupSizeX, 8u);
    this->limits.maxComputeWorkgroupSizeY =
        std::max(this->limits.maxComputeWorkgroupSizeY, 8u);
    this->limits.maxComputeWorkgroupSizeZ =
        std::max(this->limits.maxComputeWorkgroupSizeZ, 1u);

    this->limits.maxComputeInvocationsPerWorkgroup =
        std::max(this->limits.maxComputeInvocationsPerWorkgroup, 64u);

    // Dispatch grid (leave adapter default — you depend on it)
    // maxComputeWorkgroupsPerDimension unchanged
}

//================================//
void WgpuBundle::Resize(int newWidth, int newHeight)
{
    this->currentWidth = newWidth;
    this->currentHeight = newHeight;
    this->ConfigureSurface();

    this->resizeFlag = true;
}