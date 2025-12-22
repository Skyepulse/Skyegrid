#ifndef VOXELMANAGER_HPP
#define VOXELMANAGER_HPP

#include "../includes/Rendering/Pipelines/pipelines.hpp"
#include <cstdint>
#include <vector>
#include <atomic>

const int MAX_FEEDBACK = 8192;
const uint32_t MAX_GPU_BRICKS = 8192;
const int COLOR_BYTES_PER_BRICK = 2048; // 8x8x8 voxels, 1 byte per voxel (RGB packed), aligned to 2048 bytes
//================================//
struct ColorRGB
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct BrickMapCPU
{
    uint64_t occupancy[8];
    ColorRGB colors[512];
    ColorRGB lodColor;

    bool dirty = false;
    bool onGPU = false;
    uint32_t gpuBrickIndex = UINT32_MAX;
};

struct BrickGridCell
{
    // Bit layout (example):
    // [31]     : resident flag
    // [30]     : requested flag (GPU feedback)
    // [29:24]  : unused
    // [23:0]   : brick pool index OR packed LOD color
    uint32_t pointer;
};

struct GPUBrick
{
    uint32_t brickIndex;
};

struct Feedback {
    std::atomic<uint32_t> count;
    uint32_t indices[MAX_FEEDBACK];
};

struct UploadEntry
{
    uint32_t brickGridIndex;
    uint32_t gpuBrickSlot;
    uint64_t occupancy[8];
    uint8_t colors[COLOR_BYTES_PER_BRICK];
};

//================================//
class VoxelManager
{
public:
    VoxelManager(int resolution) : resolution(resolution) 
    {
        BrickResolution = resolution / 8; // A brick is 8x8x8 voxels
    };
    ~VoxelManager()
    {
        // free vectors
        brickMaps.clear();
        brickGrid.clear();
    };

    void update(wgpu::Queue& queue, wgpu::CommandEncoder& encoder);
    void readFeedback(WgpuBundle& wgpuBundle);

    void setVoxel(int x, int y, int z, bool filled, ColorRGB color);
    ColorRGB computeBrickAverageColor(const BrickMapCPU& brick);
    void initBuffers(WgpuBundle& wgpuBundle);
    void createUploadBindGroup(RenderPipelineWrapper& pipelineWrapper, WgpuBundle& wgpuBundle);

    inline uint32_t BrickGridIndex(uint32_t bx, uint32_t by, uint32_t bz)
    {
        return bx + by * BrickResolution + bz * BrickResolution * BrickResolution;
    }

     //CPU storage
    std::vector<BrickMapCPU> brickMaps;
    std::vector<BrickGridCell> brickGrid;

    //GPU storage
    wgpu::Buffer brickGridBuffer;
    wgpu::Buffer brickPoolBuffer;
    wgpu::Buffer colorPoolBuffer;

    wgpu::Buffer feedbackCountBuffer;
    wgpu::Buffer CPUfeedbackCountBuffer;
    wgpu::Buffer feedbackCountRESET;

    wgpu::Buffer feedbackIndicesBuffer;
    wgpu::Buffer CPUfeedbackIndicesBuffer;

    wgpu::Buffer uploadBuffer;
    wgpu::Buffer CPUuploadBuffer;

    std::vector<uint32_t> feedbackRequests;
    std::vector<uint32_t> freeBrickSlots;

    uint32_t pendingUploadCount = 0;

private:
    int resolution; 
    int BrickResolution;
};

#endif 