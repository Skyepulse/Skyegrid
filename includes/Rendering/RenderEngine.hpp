#ifndef RENDER_ENGINE_HPP
#define RENDER_ENGINE_HPP

#include "wgpuBundle.hpp"
#include "wgpuHelpers.hpp"
#include "Pipelines/pipelines.hpp"
#include <Eigen/Core>
#include "Camera/Camera.hpp"
#include "../constants.hpp"
#include <iostream>

//================================//
struct RenderInfo
{
    uint32_t width;
    uint32_t height;
    double time;
    bool resizeNeeded;
};

//================================//
struct VoxelParameters
{
    Eigen::Matrix4f pixelToRay;
    Eigen::Vector3f cameraOrigin;
    float _pad0;
    uint32_t voxelResolution;
    float time;
    float _pad1[2];
};

//================================//
class RenderEngine
{
public:
    RenderEngine(WgpuBundle* bundle)
    {
        std::cout << "[RenderEngine] Initializing Render Engine...\n";
        // Create Debug Pipeline
        this->wgpuBundle = bundle;
        
        // Create Camera
        WindowFormat windowFormat = bundle->GetWindowFormat();
        this->camera = std::make_unique<Camera>(Eigen::Vector2f(static_cast<float>(windowFormat.width), static_cast<float>(windowFormat.height)));

        CreateRenderPipelineDebug(*this->wgpuBundle, this->debugPipeline);
        CreateComputeVoxelPipeline(*this->wgpuBundle, this->computeVoxelPipeline);
        CreateBlitVoxelPipeline(*this->wgpuBundle, this->blitVoxelPipeline);

        std::cout << "[RenderEngine] Render Engine initialized successfully.\n";

        this->PackVoxelDataToGPU();
    }
    ~RenderEngine() = default;

    void Render(void* userData);
    void RenderDebug(void* userData);

    Camera* GetCamera() { return this->camera.get(); }

private:

    void RebuildVoxelPipelineResources(const RenderInfo& renderInfo);
    void PackVoxelDataToGPU();
    void SetPackedVoxel(uint32_t x, uint32_t y, uint32_t z, bool on);

    RenderPipelineWrapper debugPipeline;
    RenderPipelineWrapper computeVoxelPipeline;
    RenderPipelineWrapper blitVoxelPipeline;
    WgpuBundle* wgpuBundle;

    bool resizePending = true;
    std::unique_ptr<Camera> camera;

    std::vector<uint32_t> texelInfo;
};

#endif // RENDER_ENGINE_HPP