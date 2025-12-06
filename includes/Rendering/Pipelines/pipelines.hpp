#ifndef PIPELINES_HPP
#define PIPELINES_HPP

#include "../wgpuBundle.hpp"
#include "../wgpuHelpers.hpp"

//================================//
struct RenderPipelineWrapper
{
    wgpu::RenderPipeline pipeline;
    wgpu::PipelineLayout pipelineLayout;

    wgpu::BindGroup bindGroup;
    wgpu::BindGroupLayout bindGroupLayout;

    wgpu::ShaderModule shaderModule;

    int init = -1;
};


//================================//
void CreateRenderPipelineDebug(WgpuBundle& wgpuBundle, RenderPipelineWrapper& pipelineWrapper)
{
    // Code in ../../../src/Shaders/debugShader.wgsl
    std::string shaderCode;
    if (getShaderCodeFromFile("Shaders/debug.wgsl", shaderCode) < 0)
    {
        throw std::runtime_error(
            "[PIPELINES] Failed to load debug shader code from path: " +
            (getExecutableDirectory() / "Shaders/debug.wgsl").string()
        );
    }

    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = shaderCode.c_str();

    wgpu::ShaderModuleDescriptor shaderModuleDesc{};
    shaderModuleDesc.nextInChain = &wgsl;

    pipelineWrapper.shaderModule = wgpuBundle.GetDevice().CreateShaderModule(&shaderModuleDesc);
    if (!pipelineWrapper.shaderModule)
    {
        throw std::runtime_error("[PIPELINES] Failed to create debug shader module.");
    }

    wgpu::ColorTargetState colorTargetState{};
    colorTargetState.format = wgpuBundle.GetSwapchainFormat();

    wgpu::FragmentState fragmentState{};
    fragmentState.module = pipelineWrapper.shaderModule;
    fragmentState.entryPoint = "f";
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTargetState;

    wgpu::VertexState vertexState{};
    vertexState.module = pipelineWrapper.shaderModule;
    vertexState.entryPoint = "v";

    wgpu::RenderPipelineDescriptor pipelineDesc{};
    pipelineDesc.vertex = vertexState;
    pipelineDesc.fragment = &fragmentState;

    pipelineWrapper.pipeline = wgpuBundle.GetDevice().CreateRenderPipeline(&pipelineDesc);
    if (!pipelineWrapper.pipeline)
    {
        throw std::runtime_error("[PIPELINES] Failed to create debug render pipeline.");
    }

    pipelineWrapper.init = 1;
}

#endif // PIPELINES_HPP