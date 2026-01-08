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

    if (this->window != nullptr)
    {
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
    }

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

    this->supportsTimestampQuery = this->adapter.HasFeature(wgpu::FeatureName::TimestampQuery);

    // Device
    wgpu::DeviceDescriptor deviceDesc{};
    deviceDesc.requiredLimits = &this->limits;
    deviceDesc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType errorType, wgpu::StringView message)
        {
            std::cout << "[wgpuDevice] Uncaptured error: " << message << std::endl;
        }
    );

    wgpu::FeatureName requiredDeviceFeatures[] = {
        wgpu::FeatureName::TimestampQuery,
    };

    if (this->supportsTimestampQuery)
    {
        std::cout << "[wgpuBundle][Init] Timestamp query supported, enabling feature.\n";
        deviceDesc.requiredFeatures = requiredDeviceFeatures;
        deviceDesc.requiredFeatureCount = 1;
    }
    else
    {
        std::cout << "[wgpuBundle][Init] Timestamp query not supported, GPU timing unavailable.\n";
        deviceDesc.requiredFeatureCount = 0;
        deviceDesc.requiredFeatures = nullptr;
    }
    if (wgpuCreateDevice(this->instance, this->adapter, this->device, &deviceDesc) < 0)
        throw std::runtime_error("Failed to create WebGPU device.");

    wgpu::AdapterInfo info;
    adapter.GetInfo(&info);
    std::cout << "[wgpuBundle][Init] Using adapter: " << info.description << std::endl;
    std::cout << "[wgpuBundle][Init] Using device: " << info.device << std::endl;
    std::cout << "[wgpuBundle][Init] Device limits: " << std::endl;
    std::cout << "  - Max buffer size: " << this->limits.maxBufferSize << std::endl;
    std::cout << "  - Max storage buffer binding size: " << this->limits.maxStorageBufferBindingSize << std::endl;
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
    if (!this->surface)
        return;
        
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
    this->limits.maxStorageTexturesPerShaderStage =
        std::max(this->limits.maxStorageTexturesPerShaderStage, 2u);

    this->limits.maxTextureDimension2D =
        std::max(this->limits.maxTextureDimension2D,
                 static_cast<uint32_t>(
                     std::max(MAXIMUM_WINDOW_WIDTH, MAXIMUM_WINDOW_HEIGHT)));

    this->limits.maxUniformBuffersPerShaderStage =
        std::max(this->limits.maxUniformBuffersPerShaderStage, 1u);
    this->limits.maxUniformBufferBindingSize =
        std::max(static_cast<uint32_t>(this->limits.maxUniformBufferBindingSize), 256u);

    this->limits.maxComputeWorkgroupSizeX =
        std::max(this->limits.maxComputeWorkgroupSizeX, 8u);
    this->limits.maxComputeWorkgroupSizeY =
        std::max(this->limits.maxComputeWorkgroupSizeY, 8u);
    this->limits.maxComputeWorkgroupSizeZ =
        std::max(this->limits.maxComputeWorkgroupSizeZ, 1u);

    this->limits.maxComputeInvocationsPerWorkgroup =
        std::max(this->limits.maxComputeInvocationsPerWorkgroup, 128u);
}

//================================//
void WgpuBundle::Resize(int newWidth, int newHeight)
{
    this->currentWidth = newWidth;
    this->currentHeight = newHeight;
    this->ConfigureSurface();

    std::cout << "[wgpuBundle] Window resized to " << newWidth << "x" << newHeight << std::endl;
    this->resizeFlag = true;
}

//================================//
void WgpuBundle::SafeCreateBuffer(const wgpu::BufferDescriptor* descriptor, wgpu::Buffer& outBuffer)
{
    const uint64_t bufferRequestedSize = descriptor->size;
    if (bufferRequestedSize > this->limits.maxBufferSize)
    {
        std::cout << "[WgpuBundle] Requested buffer size (" << bufferRequestedSize
                    << ") exceeds device maxBufferSize limit (" << this->limits.maxBufferSize << ")." << std::endl;
        throw std::runtime_error("[WgpuBundle] Requested buffer size exceeds device limits.");
    }

    outBuffer = this->device.CreateBuffer(descriptor);
    if (!outBuffer)
    {
        std::cout << "[WgpuBundle] Failed to create buffer of size " << bufferRequestedSize << "." << std::endl;
        throw std::runtime_error("[WgpuBundle] Failed to create buffer.");
    }
}