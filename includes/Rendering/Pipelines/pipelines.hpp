#ifndef PIPELINES_HPP
#define PIPELINES_HPP

#include "../wgpuBundle.hpp"
#include "../wgpuHelpers.hpp"

//================================//
struct RenderPipelineWrapper
{
    wgpu::RenderPipeline pipeline;
    wgpu::ComputePipeline computePipeline;
    wgpu::PipelineLayout pipelineLayout;

    wgpu::BindGroup bindGroup;
    wgpu::BindGroupLayout bindGroupLayout;

    wgpu::ShaderModule shaderModule;

    int init = -1;
    bool isCompute = false;

    std::vector<wgpu::Buffer>   associatedBuffers;
    std::vector<wgpu::Buffer>   associatedUniforms;
    std::vector<wgpu::Texture>  associatedTextures;
    std::vector<wgpu::TextureView> associatedTextureViews;

    std::vector<size_t>         bufferSizes;
    std::vector<size_t>         uniformSizes;
    std::vector<size_t>         textureSizes;

    // Small method to call and assert initialization
    void AssertInitialized()
    {
        AssertConsistent();
        if (this->init != 1)
        {
            throw std::runtime_error("[PIPELINES] Attempted to use uninitialized pipeline.");
        }
    }

    // Small method to check consistency of compute pipeline
    void AssertConsistent()
    {
        if (!this->isCompute && this->pipeline.Get() == nullptr)
        {
            throw std::runtime_error("[PIPELINES] Attempted to use RenderPipelineWrapper as render pipeline when it is marked as compute.");
        }
        if (this->isCompute && this->computePipeline.Get() == nullptr)
        {
            throw std::runtime_error("[PIPELINES] Attempted to use RenderPipelineWrapper as compute pipeline when it is not marked as such.");
        }
    }
};


//================================//
void CreateRenderPipelineDebug(WgpuBundle& wgpuBundle, RenderPipelineWrapper& pipelineWrapper);
void CreateComputeVoxelPipeline(WgpuBundle& wgpuBundle, RenderPipelineWrapper& pipelineWrapper);

void InitComputeVoxelPipelineResources(RenderPipelineWrapper& pipelineWrapper, size_t voxelCount, size_t voxelParamSize);

#endif // PIPELINES_HPP