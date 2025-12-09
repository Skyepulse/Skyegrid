#ifndef RENDER_ENGINE_HPP
#define RENDER_ENGINE_HPP

#include "wgpuBundle.hpp"
#include "wgpuHelpers.hpp"
#include "Pipelines/pipelines.hpp"
#include <Eigen/Core>
#include "Camera/Camera.hpp"
#include "../constants.hpp"

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
        // Create Debug Pipeline
        this->wgpuBundle = bundle;
        
        // Create Camera
        WindowFormat windowFormat = bundle->GetWindowFormat();
        this->camera = std::make_unique<Camera>(Eigen::Vector2f(static_cast<float>(windowFormat.width), static_cast<float>(windowFormat.height)));
        
        this->camera->SetFov(45.0f);
        float r = static_cast<float>(MAXIMUM_VOXEL_RESOLUTION);

        this->camera->SetPosition(Eigen::Vector3f(r / 2.0f, r / 2.0f, -r * 1.5f));
        this->camera->LookAtPoint(Eigen::Vector3f(r / 2.0f, r / 2.0f, r / 2.0f));
        this->camera->ValidatePixelToRayMatrix();

        CreateRenderPipelineDebug(*this->wgpuBundle, this->debugPipeline);
        CreateComputeVoxelPipeline(*this->wgpuBundle, this->computeVoxelPipeline);
        CreateBlitVoxelPipeline(*this->wgpuBundle, this->blitVoxelPipeline);

        voxelDataCache.resize(MAXIMUM_VOXEL_RESOLUTION);
        for (int x = 0; x < MAXIMUM_VOXEL_RESOLUTION; x++)
        {
            voxelDataCache[x].resize(MAXIMUM_VOXEL_RESOLUTION);
            for (int y = 0; y < MAXIMUM_VOXEL_RESOLUTION; y++)
            {
                voxelDataCache[x][y].resize(MAXIMUM_VOXEL_RESOLUTION, 0);
            }
        }

        // Cube edges
        for (int x = 0; x < MAXIMUM_VOXEL_RESOLUTION; ++x)
        for (int y = 0; y < MAXIMUM_VOXEL_RESOLUTION; ++y)
        for (int z = 0; z < MAXIMUM_VOXEL_RESOLUTION; ++z)
        {
            bool edge =
                // edges parallel to X
                ((y == 0 || y == MAXIMUM_VOXEL_RESOLUTION - 1) &&
                (z == 0 || z == MAXIMUM_VOXEL_RESOLUTION - 1)) ||

                // edges parallel to Y
                ((x == 0 || x == MAXIMUM_VOXEL_RESOLUTION - 1) &&
                (z == 0 || z == MAXIMUM_VOXEL_RESOLUTION - 1)) ||

                // edges parallel to Z
                ((x == 0 || x == MAXIMUM_VOXEL_RESOLUTION - 1) &&
                (y == 0 || y == MAXIMUM_VOXEL_RESOLUTION - 1));

            voxelDataCache[x][y][z] = edge ? 1 : 0;
        }

        // Cube in the exact middle
        for (int x = MAXIMUM_VOXEL_RESOLUTION / 4; x < 3 * MAXIMUM_VOXEL_RESOLUTION / 4; ++x)
        for (int y = MAXIMUM_VOXEL_RESOLUTION / 4; y < 3 * MAXIMUM_VOXEL_RESOLUTION / 4; ++y)
        for (int z = MAXIMUM_VOXEL_RESOLUTION / 4; z < 3 * MAXIMUM_VOXEL_RESOLUTION / 4; ++z)
        {
            voxelDataCache[x][y][z] = 1;
        }

        this->PackVoxelDataToGPU();
    }
    ~RenderEngine() = default;

    void Render(void* userData);

    Camera* GetCamera() { return this->camera.get(); }

private:

    void RebuildVoxelPipelineResources(const RenderInfo& renderInfo);
    void PackVoxelDataToGPU();

    RenderPipelineWrapper debugPipeline;
    RenderPipelineWrapper computeVoxelPipeline;
    RenderPipelineWrapper blitVoxelPipeline;
    WgpuBundle* wgpuBundle;

    bool resizePending = true;
    std::unique_ptr<Camera> camera;

    std::vector<std::vector<std::vector<uint8_t>>> voxelDataCache;
};

#endif // RENDER_ENGINE_HPP