#ifndef VOXELMANAGER_HPP
#define VOXELMANAGER_HPP

#include "../includes/Rendering/Pipelines/pipelines.hpp"
#include <cstdint>
#include <vector>
#include <atomic>
#include <mutex>
#include <iostream>

class VoxelManager; // Forward declaration

const int MAX_FEEDBACK = 8192;

// Max bricks is max index that we can pack in 24 bits, which is 2^24 - 1
const int MAX_BRICKS = 16777215;
const int COLOR_BYTES_PER_BRICK = 2048; // 8x8x8 voxels, 1 byte per voxel (RGB packed), aligned to 2048 bytes
//================================//
struct ColorRGB
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t _pad;
};

struct BrickMapCPU
{
    // first slice in z is occupancy[0] for first half, occupancy[1] for second half
    uint32_t occupancy[16]; // we should interpret 8 slices of 32 + 32 bits each = 512 voxels
    ColorRGB colors[512];
    ColorRGB lodColor;

    bool dirty = false;
    bool onGPU = false;
    uint32_t gpuBrickIndex = UINT32_MAX;
};

struct BrickGridCell
{
    // [23:0]   : pointer / index or LOD in case unloaded (r, g, b)
    // [31]     : resident flag
    // [30]     : requested flag
    // [29]     : unloaded flag
    // [28:24]  : unused
    uint32_t pointer;
};

struct Feedback 
{
    std::atomic<uint32_t> count;
    uint32_t indices[MAX_FEEDBACK];
};

struct UploadEntry
{
    uint32_t gpuBrickSlot;
    uint32_t occupancy[16];
    ColorRGB colors[512];
};

//================================//
class VoxelManager
{
public:
    VoxelManager(int resolution) : resolution(resolution) 
    {
        BrickResolution = resolution / 8; // A brick is 8x8x8 voxels
        const int num_bricks = BrickResolution * BrickResolution * BrickResolution;

        if (num_bricks > MAX_BRICKS)
        {
            throw std::runtime_error("[VoxelManager] Voxel resolution too high, exceeds maximum number of bricks that can be addressed.");
        }
    };
    ~VoxelManager()
    {
        // free vectors
        brickMaps.clear();
        brickGrid.clear();
    };

    void update(WgpuBundle& wgpuBundle, const wgpu::Queue& queue, const wgpu::CommandEncoder& encoder);
    void readFeedback(wgpu::Future& outFuture);
    void prepareFeedback(const wgpu::Queue& queue, const wgpu::CommandEncoder& encoder);

    void setVoxel(int x, int y, int z, bool filled, ColorRGB color);
    ColorRGB computeBrickAverageColor(const BrickMapCPU& brick);
    void initBuffers(WgpuBundle& wgpuBundle);
    void createUploadBindGroup(RenderPipelineWrapper& pipelineWrapper, WgpuBundle& wgpuBundle);
    void startOfFrame();

    bool remapUploadBuffer(wgpu::Future& outFuture);

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
    wgpu::Buffer feedbackCountRESET;
    wgpu::Buffer feedbackIndicesBuffer;
    wgpu::Buffer CPUfeedbackBuffer; // This is a mapRead buffer containing both count and then indices

    wgpu::Buffer uploadBuffer;
    wgpu::Buffer CPUuploadBuffer;
    wgpu::Buffer uploadCountUniform;

    std::vector<uint32_t> feedbackRequests;
    std::vector<uint32_t> freeBrickSlots;
    std::vector<uint32_t> dirtyBrickIndices;

    uint32_t pendingUploadCount = 0;
    uint32_t lastBrickIndex = 0; // DEBUG

private:
    bool requestRemap = false;

    int resolution; 
    int BrickResolution;
};

#endif 