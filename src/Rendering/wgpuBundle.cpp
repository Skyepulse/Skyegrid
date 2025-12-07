#include "../../includes/Rendering/wgpuBundle.hpp"
#include "../../includes/Rendering/wgpuHelpers.hpp"
#include "../../includes/constants.hpp"

#include <vector>
#include <iostream>

//================================//
WgpuBundle::WgpuBundle(WindowFormat windowFormat) : window(windowFormat.window), currentWidth(windowFormat.width), currentHeight(windowFormat.height)
{
    this->ComputeLimits();
    this->InitializeInstance();
    this->surface = wgpu::glfw::CreateSurfaceForWindow(instance, window);
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

    surface.Configure(&config);
}

//================================//
void WgpuBundle::ComputeLimits()
{
    this->limits = {};

    this->limits.maxStorageBuffersPerShaderStage = 0;
    this->limits.maxStorageBufferBindingSize    = 0;

    this->limits.maxStorageTexturesPerShaderStage = 2;

    size_t maxTextureDimension3D = static_cast<size_t>(MAXIMUM_VOXEL_RESOLUTION / 4);
    size_t maxTextureDimension2D = static_cast<size_t>(std::max(MAXIMUM_WINDOW_WIDTH, MAXIMUM_WINDOW_HEIGHT));
    this->limits.maxTextureDimension3D = static_cast<uint32_t>(maxTextureDimension3D);
    this->limits.maxTextureDimension2D = static_cast<uint32_t>(maxTextureDimension2D);

    this->limits.maxUniformBuffersPerShaderStage = 1;
    this->limits.maxUniformBufferBindingSize = 256;

    // for now (8, 8, 1) workgroup size with 64 invocations
    this->limits.maxComputeWorkgroupSizeX = 8;
    this->limits.maxComputeWorkgroupSizeY = 8;
    this->limits.maxComputeWorkgroupSizeZ = 1;
    this->limits.maxComputeInvocationsPerWorkgroup = 64;

    this->limits.maxComputeWorkgroupsPerDimension = 65535;
}