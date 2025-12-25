#ifndef RENDER_ENGINE_HPP
#define RENDER_ENGINE_HPP

#include "wgpuBundle.hpp"
#include "wgpuHelpers.hpp"
#include "../VoxelManager.hpp"
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
    uint32_t maxColorBufferSize; // Used to derive which color pool to read from
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
        
        // Create Voxel Manager
        this->voxelManager = std::make_unique<VoxelManager>(*this->wgpuBundle, static_cast<int>(MAXIMUM_VOXEL_RESOLUTION));
        
        // Create Camera
        WindowFormat windowFormat = bundle->GetWindowFormat();
        this->camera = std::make_unique<Camera>(Eigen::Vector2f(static_cast<float>(windowFormat.width), static_cast<float>(windowFormat.height)));

        std::cout << "[RenderEngine] Creating Pipelines...\n";
        CreateRenderPipelineDebug(*this->wgpuBundle, this->debugPipeline);
        CreateComputeVoxelPipeline(*this->wgpuBundle, this->computeVoxelPipeline, MAX_COLOR_POOLS);
        CreateComputeUploadVoxelPipeline(*this->wgpuBundle, this->computeUploadVoxelPipeline, MAX_COLOR_POOLS);
        CreateBlitVoxelPipeline(*this->wgpuBundle, this->blitVoxelPipeline);
        std::cout << "[RenderEngine] Creating pipelines completed.\n";

        std::cout << "[RenderEngine] Initializing Voxel Manager...\n";
        this->voxelManager->initBuffers(*this->wgpuBundle);
        this->voxelManager->createUploadBindGroup(this->computeUploadVoxelPipeline, *this->wgpuBundle);

        std::cout << "[RenderEngine] Render Engine initialized successfully.\n";
    }
    ~RenderEngine() = default;

    void Render(void* userData);
    void RenderDebug(void* userData);

    Camera* GetCamera() { return this->camera.get(); }

private:

    void RebuildVoxelPipelineResources(const RenderInfo& renderInfo);

    RenderPipelineWrapper debugPipeline;
    RenderPipelineWrapper computeVoxelPipeline;
    RenderPipelineWrapper computeUploadVoxelPipeline;
    RenderPipelineWrapper blitVoxelPipeline;
    WgpuBundle* wgpuBundle;

    bool resizePending = true;
    std::unique_ptr<Camera> camera;

    std::vector<uint32_t> texelInfo;
    std::unique_ptr<VoxelManager> voxelManager;
};

#endif // RENDER_ENGINE_HPP