#ifndef RENDER_ENGINE_HPP
#define RENDER_ENGINE_HPP

#include "wgpuBundle.hpp"
#include "wgpuHelpers.hpp"
#include "Pipelines/pipelines.hpp"

//================================//
struct RenderInfo
{
    int width;
    int height;
    double time;
};

//================================//
struct VoxelParameters
{
    uint32_t voxelDimensions[3];
    float    time;
};

//================================//
class RenderEngine
{
public:
    RenderEngine(WgpuBundle* bundle)
    {
        this->wgpuBundle = bundle;
        CreateRenderPipelineDebug(*this->wgpuBundle, this->debugPipeline);
        CreateComputeVoxelPipeline(*this->wgpuBundle, this->computeVoxelPipeline);
        CreateBlitVoxelPipeline(*this->wgpuBundle, this->blitVoxelPipeline);
    };
    ~RenderEngine() = default;

    void Render(void* userData);

private:
    RenderPipelineWrapper debugPipeline;
    RenderPipelineWrapper computeVoxelPipeline;
    RenderPipelineWrapper blitVoxelPipeline;
    WgpuBundle* wgpuBundle;

    bool resizePending = true;
};

#endif // RENDER_ENGINE_HPP