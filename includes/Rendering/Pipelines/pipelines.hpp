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
    std::vector<wgpu::Sampler>  associatedSamplers;

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
        // Consistent with isCompute flag
        if (!this->isCompute && this->pipeline.Get() == nullptr)
        {
            throw std::runtime_error("[PIPELINES] Attempted to use RenderPipelineWrapper as render pipeline when it is marked as compute.");
        }
        if (this->isCompute && this->computePipeline.Get() == nullptr)
        {
            throw std::runtime_error("[PIPELINES] Attempted to use RenderPipelineWrapper as compute pipeline when it is not marked as such.");
        }

        // Consistent associated resources
        int numBuffers = static_cast<int>(this->bufferSizes.size());
        int numTextures = static_cast<int>(this->textureSizes.size());
        int numTextureViews = static_cast<int>(this->textureSizes.size());
        int numUniforms = static_cast<int>(this->uniformSizes.size());

        if (static_cast<int>(this->associatedBuffers.size()) != numBuffers)
        {
            throw std::runtime_error("[PIPELINES] Inconsistent number of associated buffers in RenderPipelineWrapper.");
        }
        if (static_cast<int>(this->associatedTextures.size()) != numTextures)
        {
            throw std::runtime_error("[PIPELINES] Inconsistent number of associated textures in RenderPipelineWrapper.");
        }
        if (static_cast<int>(this->associatedTextureViews.size()) != numTextureViews)
        {
            throw std::runtime_error("[PIPELINES] Inconsistent number of associated texture views in RenderPipelineWrapper.");
        }
        if (static_cast<int>(this->associatedUniforms.size()) != numUniforms)
        {
            throw std::runtime_error("[PIPELINES] Inconsistent number of associated uniforms in RenderPipelineWrapper.");
        }
    }
};


//================================//
void CreateRenderPipelineDebug(WgpuBundle& wgpuBundle, RenderPipelineWrapper& pipelineWrapper);
void CreateComputeVoxelPipeline(WgpuBundle& wgpuBundle, RenderPipelineWrapper& pipelineWrapper);
void CreateBlitVoxelPipeline(WgpuBundle& wgpuBundle, RenderPipelineWrapper& pipelineWrapper);

void InitComputeVoxelPipelineResources(RenderPipelineWrapper& pipelineWrapper, size_t voxelCount, size_t voxelParamSize);


#endif // PIPELINES_HPP