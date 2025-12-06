#include <webgpu/webgpu_cpp.h>
#include <dawn/webgpu_cpp_print.h>

class ComputeTest
{
    public:
        void initBuffers(const wgpu::Device& device);
        void createBindGroupLayout(const wgpu::Device& device);
        void createBindGroup(const wgpu::Device& device);
        void createComputePipeline(const wgpu::Device& device);
        void OnCompute(const wgpu::Instance& instance, const wgpu::Device& device);

        wgpu::PipelineLayout pipelineLayout;
        wgpu::ComputePipeline pipeline;
        
        wgpu::BindGroupLayout bindGroupLayout;
        wgpu::BindGroup bindGroup;

        wgpu::Buffer inputBuffer;
        wgpu::Buffer outputBuffer;
        wgpu::Buffer mapBuffer;

        size_t bufferSize = 64 * sizeof(float);
};