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

// ImGUI
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_wgpu.h"

// forward declarations
class RenderEngine;

//================================//
struct RenderInfo
{
    uint32_t width;
    uint32_t height;
    double time;
    bool resizeNeeded;
};
struct VoxelParameters
{
    Eigen::Matrix4f pixelToRay;
    Eigen::Vector3f cameraOrigin;
    uint32_t maxColorBufferSize; // Used to derive which color pool to read from
    uint32_t voxelResolution;
    float time;
    uint32_t hasColor;
    uint32_t flip; // bits 0: flipX, 1: flipY, 2: flipZ
};

struct TimingCtx 
{
    RenderEngine* engine;
    int bufferIndex;
};
struct FeedbackCtx {
    VoxelManager* vm;
    int slot;
};

//================================//
class RenderEngine
{
public:
    RenderEngine(WgpuBundle* bundle, int voxelResolution, int maxVisibleBricks)
    {
        std::cout << "[RenderEngine] Initializing Render Engine...\n";
        this->wgpuBundle = bundle;
        
        // Create Voxel Manager
        this->voxelManager = std::make_unique<VoxelManager>(*this->wgpuBundle, voxelResolution, maxVisibleBricks);
        
        // Create Camera
        WindowFormat windowFormat = bundle->GetWindowFormat();
        this->camera = std::make_unique<Camera>(Eigen::Vector2f(static_cast<float>(windowFormat.width), static_cast<float>(windowFormat.height)));
    }
    ~RenderEngine()
    {
        // ImGUI Cleanup
        ImGui_ImplWGPU_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    void Initialize()
    {
        // Initialize GPU Timing Queries
        this->InitializeGPUTimingQueries();

        std::cout << "[RenderEngine] Creating Pipelines...\n";
        CreateRenderPipelineDebug(*this->wgpuBundle, this->debugPipeline);
        CreateComputeVoxelPipeline(*this->wgpuBundle, this->computeVoxelPipeline, MAX_COLOR_POOLS);
        CreateComputeUploadVoxelPipeline(*this->wgpuBundle, this->computeUploadVoxelPipeline, MAX_COLOR_POOLS);
        CreateBlitVoxelPipeline(*this->wgpuBundle, this->blitVoxelPipeline);
        std::cout << "[RenderEngine] Creating pipelines completed.\n";

        std::cout << "[RenderEngine] Initializing Voxel Manager...\n";
        this->voxelManager->initStaticBuffers(*this->wgpuBundle);
        this->voxelManager->initDynamicBuffers(*this->wgpuBundle);
        this->voxelManager->createUploadBindGroup(this->computeUploadVoxelPipeline, *this->wgpuBundle);

        InitImGui();
        std::cout << "[RenderEngine] Render Engine initialized successfully.\n";
    }

    void Render(void* userData);
    void RenderDebug(void* userData);
    void loadFile(const std::string& filename)
    {
        this->voxelManager->loadFile(filename);
    }

    Camera* GetCamera() { return this->camera.get(); }
    int GetVoxelResolution() const { return this->voxelManager->GetVoxelResolution(); }

private:

    void InitImGui();
    void RenderImGui(wgpu::RenderPassEncoder& pass);
    void onResolutionValueChanged(int newResolution);
    void onVisibleBricksValueChanged(int newMaxVisibleBricks);

    void flipAxis(int axis); // 0, 1, 2

    void ReadFeedbacks();

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

    uint32_t flipBits = 0;

    // ImGUI
    int resolutionSliderValue = -1;
    int previousResolutionValue = -1;
    int visibleBricksSliderValue = -1;
    int previousVisibleBricksValue = -1;
    int resolutionDigitBoxValue = -1;
    int visibleBricksDigitBoxValue = -1;
    int flipXCheckbox = 0;
    int flipYCheckbox = 0;
    int flipZCheckbox = 0;

    // Timing info
    float cpuFrameTimeMS = 0.0f;
    std::vector<float> cpuFrameAccumulator;

    float gpuFrameTimeRayTraceMs = 0.0f;
    float gpuFrameTimeUploadMs = 0.0f;
    float gpuFrameTimeBlitMs = 0.0f;
    std::vector<float> gpuFrameRayTraceAccumulator;
    std::vector<float> gpuFrameUploadAccumulator;
    std::vector<float> gpuFrameBlitAccumulator;

    wgpu::QuerySet gpuTimingQuerySet;
    wgpu::Buffer gpuTimingResolveBuffer;
    wgpu::Buffer gpuTimingReadbackBuffers[2];

    void InitializeGPUTimingQueries();
    void ReadTimingQueries();
    bool gpuTimingMapInFlight = false;
    int currentTimingWriteBuffer = 0;
};

#endif // RENDER_ENGINE_HPP