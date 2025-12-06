#include <iostream>
#include <GLFW/glfw3.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

#include <webgpu/webgpu_cpp.h>
#include <dawn/webgpu_cpp_print.h>
#include <webgpu/webgpu_glfw.h>

#include <Eigen/Core>

#include "src/computeTest.h"

const uint32_t kWidth = 512;
const uint32_t kHeight = 512;

using namespace std;
using namespace wgpu;

Instance instance;

Adapter adapter;
Device device;

Surface surface;
TextureFormat swapchainFormat;

RenderPipeline pipeline;

ComputeTest computeTest;

const char ShaderCode[] = R"(
    @vertex
    fn v(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> 
    {
        var pos = array<vec2<f32>, 3>(
            vec2<f32>(0.0, 0.5),
            vec2<f32>(-0.5, -0.5),
            vec2<f32>(0.5, -0.5)
        );
        return vec4<f32>(pos[i], 0.0, 1.0);
    }

    @fragment 
    fn f() -> @location(0) vec4<f32> 
    {
        return vec4<f32>(0.0, 0.5, 0.5, 1.0);
    }
)";

void InitWebGPU()
{
    computeTest = ComputeTest();

    // Create instance
    static const auto kTimeWaitAny = InstanceFeatureName::TimedWaitAny;
    InstanceDescriptor descriptor;
    descriptor.requiredFeatureCount = 1;
    descriptor.requiredFeatures = &kTimeWaitAny;

    Limits limits{};
    limits.maxStorageBuffersPerShaderStage = 2;
    limits.maxStorageBufferBindingSize = computeTest.bufferSize;
    limits.maxComputeWorkgroupSizeX = 32;
    limits.maxComputeWorkgroupSizeY = 1;
    limits.maxComputeWorkgroupSizeZ = 1;
    limits.maxComputeInvocationsPerWorkgroup = 32;
    limits.maxComputeWorkgroupsPerDimension = 2;

    instance = CreateInstance(&descriptor);

    RequestAdapterOptions options{};
    options.backendType = BackendType::Null;

    // Get adapter
    Future f1 = instance.RequestAdapter(
        nullptr,
        CallbackMode::WaitAnyOnly,
        [](RequestAdapterStatus status, Adapter a, StringView message)
        {
            if (status != RequestAdapterStatus::Success)
            {
                cout << "Failed to get adapter: " << message << endl;
            }
            adapter = move(a);
        }
    );
    instance.WaitAny(f1, UINT64_MAX); // Timeout to max

    DeviceDescriptor deviceDesc{};
    deviceDesc.requiredLimits = &limits;
    deviceDesc.SetUncapturedErrorCallback(
        [](const Device&, ErrorType errorType, StringView message)
        {
            cout << "Uncaptured error: " << message << endl;
        }
    );

    // Get Device
    Future f2 = adapter.RequestDevice(
        &deviceDesc,
        CallbackMode::WaitAnyOnly,
        [](RequestDeviceStatus status, Device d, StringView message)
        {
            if (status != RequestDeviceStatus::Success)
            {
                cout << "Failed to get device: " << message << endl;
            }
            device = move(d);
        }
    );
    instance.WaitAny(f2, UINT64_MAX); // Timeout to max

    computeTest.initBuffers(device);
    computeTest.createBindGroupLayout(device);
    computeTest.createBindGroup(device);
    computeTest.createComputePipeline(device);
}

void ConfigureSurface()
{
    SurfaceCapabilities capabilities;
    surface.GetCapabilities(adapter, &capabilities);
    swapchainFormat = capabilities.formats[0];

    SurfaceConfiguration config{};
    config.device = device;
    config.format = swapchainFormat;
    config.width = kWidth;
    config.height = kHeight;

    surface.Configure(&config);
}

void CreateRenderPipeline()
{
    ShaderSourceWGSL wgsl{};
    wgsl.code = ShaderCode;

    ShaderModuleDescriptor shaderDesc{};
    shaderDesc.nextInChain = &wgsl;

    ShaderModule shaderModule = device.CreateShaderModule(&shaderDesc);
    
    ColorTargetState colorTargetState{};
    colorTargetState.format = swapchainFormat;

    FragmentState fragmentState{};
    fragmentState.module = shaderModule;
    fragmentState.targetCount = 1; // This represents the number of color attachments
    fragmentState.targets = &colorTargetState;

    VertexState vertexState{};
    vertexState.module = shaderModule;

    RenderPipelineDescriptor descriptor{};
    descriptor.vertex = vertexState;
    descriptor.fragment = &fragmentState;

    pipeline = device.CreateRenderPipeline(&descriptor);
}

void InitGraphics()
{
    ConfigureSurface();
    CreateRenderPipeline();
}

void Render()
{
    SurfaceTexture surfaceTexture;
    surface.GetCurrentTexture(&surfaceTexture);

    RenderPassColorAttachment colorAttachment{};
    colorAttachment.view = surfaceTexture.texture.CreateView();
    colorAttachment.loadOp = LoadOp::Clear;
    colorAttachment.clearValue = {0.3f, 0.3f, 0.3f, 1.0f};
    colorAttachment.storeOp = StoreOp::Store;

    RenderPassDescriptor renderPassDesc{};
    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &colorAttachment;

    CommandEncoder encoder = device.CreateCommandEncoder();
    RenderPassEncoder pass = encoder.BeginRenderPass(&renderPassDesc);

    pass.SetPipeline(pipeline);
    pass.Draw(3); // Draw 3 vertices
    pass.End();
    CommandBuffer commandBuffer = encoder.Finish();

    device.GetQueue().Submit(1, &commandBuffer);
}

int main()
{
    InitWebGPU();

    if (!glfwInit())
    {
        cerr << "Failed to initialize GLFW" << endl;
        return -1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(kWidth, kHeight, "Skyegrid", nullptr, nullptr);

    if (!window)
    {
        cerr << "Failed to create GLFW window" << endl;
        glfwTerminate();
        return -1;
    }

    surface = wgpu::glfw::CreateSurfaceForWindow(instance, window);
    InitGraphics();

    AdapterInfo info;
    adapter.GetInfo(&info);
    cout << "Using adapter: " << info.description << endl;
    cout << "Using device: " << info.device << endl;
    cout << "Backend: " << static_cast<uint32_t>(info.backendType) << endl;

    cout << "Testing eigen..." << endl;
    Eigen::MatrixXd mat(2,2);
    mat(0,0) = 3;
    mat(1,0) = 2.5;
    mat(0,1) = -1;
    mat(1,1) = mat(1,0) + mat(0,1);
    cout << mat << endl;
    cout << "Starting main loop..." << endl;

    cout << "Running compute test..." << endl;
    computeTest.OnCompute(instance, device);
    cout << "Compute test completed." << endl;

#if defined(__EMSCRIPTEN__)
  emscripten_set_main_loop(Render, 0, true);
#else
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        Render();
        surface.Present(); // Swapchain present
        instance.ProcessEvents();
    }
#endif

    glfwDestroyWindow(window);
    glfwTerminate();

    cout << "Bye, Skyegrid!" << endl;

    return 0;
}