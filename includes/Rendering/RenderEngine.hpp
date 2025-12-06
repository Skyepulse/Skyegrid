#ifndef RENDER_ENGINE_HPP
#define RENDER_ENGINE_HPP

#include "wgpuBundle.hpp"
#include "wgpuHelpers.hpp"
#include "Pipelines/pipelines.hpp"

//================================//
struct RenderInfo
{
    int width;
};

//================================//
class RenderEngine
{
public:
    RenderEngine(WgpuBundle* bundle)
    {
        this->wgpuBundle = bundle;
        CreateRenderPipelineDebug(*this->wgpuBundle, this->debugPipeline);
    };
    ~RenderEngine() = default;

    void Render(void* userData);

private:
    RenderPipelineWrapper debugPipeline;
    WgpuBundle* wgpuBundle;
};

#endif // RENDER_ENGINE_HPP