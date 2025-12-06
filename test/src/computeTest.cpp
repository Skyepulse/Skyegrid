#include "computeTest.h"
#include <iostream>

const char ShaderCode[] = R"(

    @group(0) @binding(0) var<storage,read> inputBuffer: array<f32,64>;
    @group(0) @binding(1) var<storage,read_write> outputBuffer: array<f32,64>;

    fn f(x: f32) -> f32 
    {
        return x * x + 2.0 * x + 1.0;
    }

    @compute @workgroup_size(32)
    fn c(@builtin(global_invocation_id) id: vec3<u32>) 
    {
        // Apply the function f to the buffer element at index id.x:
        outputBuffer[id.x] = f(inputBuffer[id.x]);
    }
)";

struct ComputeCallbackContext {
    wgpu::Buffer mapBuffer;
    size_t bufferSize;
    std::vector<float> inputData;
    bool* done;
};

void ComputeTest::OnCompute(const wgpu::Instance& instance, const wgpu::Device& device)
{
    wgpu::Queue queue = device.GetQueue();
    wgpu::CommandEncoderDescriptor cmdDesc{};
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder(&cmdDesc);

    wgpu::ComputePassDescriptor computePassDesc{};
    computePassDesc.timestampWrites = nullptr;
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass(&computePassDesc);

    // Write on inputbuffer
    std::vector<float> inputData(bufferSize / sizeof(float));
    for (size_t i = 0; i < inputData.size(); ++i)
    {
        inputData[i] = static_cast<float>(i) * 0.1f;
    }
    queue.WriteBuffer(inputBuffer, 0, inputData.data(), bufferSize);

    // use it
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bindGroup, 0, nullptr);

    uint32_t invocationCount = bufferSize / sizeof(float);
    uint32_t workgroupSize = 32;
    uint32_t workgroupCount = (invocationCount + workgroupSize - 1) / workgroupSize;
    pass.DispatchWorkgroups(workgroupCount, 1, 1);
    pass.End();

    encoder.CopyBufferToBuffer(outputBuffer, 0, mapBuffer, 0, bufferSize);

    wgpu::CommandBufferDescriptor commandBufferDesc{};
    wgpu::CommandBuffer commands = encoder.Finish(&commandBufferDesc);
    queue.Submit(1, &commands);

    bool done = false;

    ComputeCallbackContext* context = new ComputeCallbackContext{
        mapBuffer,
        bufferSize,
        inputData,
        &done
    };

    static auto OnMapped = [](wgpu::MapAsyncStatus status,
                              wgpu::StringView message,
                            ComputeCallbackContext* userdata)
    {
        auto* ctx = reinterpret_cast<ComputeCallbackContext*>(userdata);

        if (status == wgpu::MapAsyncStatus::Success)
        {
            const float* output =
                static_cast<const float*>(ctx->mapBuffer.GetConstMappedRange(0, ctx->bufferSize));

            for (size_t i = 0; i < ctx->inputData.size(); ++i)
            {
                std::cout << "input " << ctx->inputData[i]
                          << " -> "   << output[i] << std::endl;
            }

            ctx->mapBuffer.Unmap();
        }
        else
        {
            std::cerr << "Failed to map buffer: " << message << std::endl;
        }

        *(ctx->done) = true;
        delete ctx;
    };

    // Read back the results
    wgpu::Future f = mapBuffer.MapAsync(
        wgpu::MapMode::Read, 0, bufferSize,
        wgpu::CallbackMode::WaitAnyOnly,
        OnMapped,
        context
    );
   
    // Wait until done
    std::cout << "Waiting for compute pass to complete..." << std::endl;
    instance.WaitAny(f, UINT64_MAX);
}

void ComputeTest::createComputePipeline(const wgpu::Device& device)
{
    wgpu::ShaderSourceWGSL shaderSource{};
    shaderSource.code = ShaderCode;

    wgpu::ShaderModuleDescriptor shaderDesc{};
    shaderDesc.nextInChain = &shaderSource;

    wgpu::ShaderModule computeShaderModule = device.CreateShaderModule(&shaderDesc);

    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc{};
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = &bindGroupLayout;
    pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);

    wgpu::ComputePipelineDescriptor computePipelineDesc{};
    computePipelineDesc.layout = pipelineLayout;
    computePipelineDesc.compute.module = computeShaderModule;
    computePipelineDesc.compute.entryPoint = "c";
    pipeline = device.CreateComputePipeline(&computePipelineDesc);
}

void ComputeTest::createBindGroupLayout(const wgpu::Device& device)
{
    std::vector<wgpu::BindGroupLayoutEntry> entries(2);

    // Input buffer
    entries[0].binding = 0;
    entries[0].visibility = wgpu::ShaderStage::Compute;
    entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    // Output buffer
    entries[1].binding = 1;
    entries[1].visibility = wgpu::ShaderStage::Compute;
    entries[1].buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc{};
    bindGroupLayoutDesc.entryCount = static_cast<uint32_t>(entries.size());
    bindGroupLayoutDesc.entries = entries.data();
    bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc);
}

void ComputeTest::createBindGroup(const wgpu::Device& device)
{
    std::vector<wgpu::BindGroupEntry> entries(2);

    // Input buffer
    entries[0].binding = 0;
    entries[0].buffer = inputBuffer;
    entries[0].offset = 0;
    entries[0].size = bufferSize;

    // Output buffer
    entries[1].binding = 1;
    entries[1].buffer = outputBuffer;
    entries[1].offset = 0;
    entries[1].size = bufferSize;

    wgpu::BindGroupDescriptor bindGroupDesc{};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = static_cast<uint32_t>(entries.size());
    bindGroupDesc.entries = entries.data();

    bindGroup = device.CreateBindGroup(&bindGroupDesc);
}

void ComputeTest::initBuffers(wgpu::Device const& device)
{
    wgpu::BufferDescriptor bufferDesc;
    bufferDesc.mappedAtCreation = false;
    bufferDesc.size = bufferSize;
    bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    inputBuffer = device.CreateBuffer(&bufferDesc);

    bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    outputBuffer = device.CreateBuffer(&bufferDesc);

    bufferDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    mapBuffer = device.CreateBuffer(&bufferDesc);
}