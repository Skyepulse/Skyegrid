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
void CreateRenderPipelineDebug(WgpuBundle& wgpuBundle, RenderPipelineWrapper& pipelineWrapper);

#endif // PIPELINES_HPP