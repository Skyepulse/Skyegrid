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
const int MAX_COLOR_POOLS = 3;
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

struct UploadUniform
{
    uint32_t uploadCount;
    uint32_t maxColorBufferSize;
};

//================================//
class VoxelManager
{
public:
    VoxelManager(WgpuBundle& bundle, int resolution) : resolution(resolution) 
    {
        static_assert(sizeof(ColorRGB) == 4); // packed in UINT32

        BrickResolution = resolution / 8; // A brick is 8x8x8 voxels
        const int num_bricks = BrickResolution * BrickResolution * BrickResolution;

        if (num_bricks > MAX_BRICKS)
        {
            std::cout << "[VoxelManager] Voxel resolution too high, exceeds maximum number of bricks that can be addressed." << std::endl;
            throw std::runtime_error("[VoxelManager] Voxel resolution too high, exceeds maximum number of bricks that can be addressed.");
        }

        // Compute number of color pools needed
        uint64_t maxBufferSize = bundle.GetLimits().maxBufferSize;

        // Since we cannot split a voxel color in two, make sure we clamp the maxBufferSize to a multiple of COLOR_BYTES_PER_BRICK
        uint64_t maxColorBufferSize = (maxBufferSize / COLOR_BYTES_PER_BRICK) * COLOR_BYTES_PER_BRICK;
        this->maxColorBufferSize = static_cast<uint32_t>(maxColorBufferSize / sizeof(ColorRGB)); // in number of ColorRGB entries per buffer pool entry

        uint64_t totalColorSizeNeeded = uint64_t(num_bricks) * COLOR_BYTES_PER_BRICK; // num_bricks * 2048
        numberOfColorPools = static_cast<uint32_t>((totalColorSizeNeeded + maxColorBufferSize - 1) / maxColorBufferSize); // ceiling division
        if (numberOfColorPools <= 0)
        {
            std::cout << "[VoxelManager] No color pool buffers needed. This should not happen." << std::endl;
            throw std::runtime_error("[VoxelManager] No color pool buffers needed. This should not happen.");
        }
        if (numberOfColorPools > MAX_COLOR_POOLS)
        {
            std::cout << "[VoxelManager] Voxel resolution too high, exceeds maximum color pool buffers. Would need " << numberOfColorPools << " but max is " << MAX_COLOR_POOLS << "." << std::endl;
            // compute max possible voxel resolution with this limit
            // since the max size per pool is maxColorBufferSize, total size is MAX_COLOR_POOLS * maxColorBufferSize
            uint64_t maxTotalColorSize = uint64_t(MAX_COLOR_POOLS) * maxColorBufferSize;
            uint64_t maxBricksPossible = maxTotalColorSize / COLOR_BYTES_PER_BRICK;
            uint64_t maxVoxels = static_cast<uint64_t>(maxBricksPossible) * 512;
            // resolution is maxVoxels^(1/3)

            uint32_t maxResolution = std::pow(maxVoxels, 1.0 / 3.0);
            std::cout << "[VoxelManager] Maximum possible voxel resolution with current limits is approximately " << maxResolution << " (" << maxVoxels << " total voxels)." << std::endl;
            throw std::runtime_error("[VoxelManager] Voxel resolution too high, exceeds maximum color pool buffers.");
        }

        std::cout << "[VoxelManager] Created with voxel resolution " << resolution << " (" << resolution * resolution * resolution << " total voxels)." << std::endl;
        std::cout << "[VoxelManager] Using " << numberOfColorPools << " color pool buffers." << std::endl;
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

    std::vector<wgpu::Buffer> colorPoolBuffers;

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
    uint32_t numberOfColorPools = 0;
    uint32_t maxColorBufferSize = 0;

    uint64_t lastBrickIndex = 0; // DEBUG

private:
    bool requestRemap = false;

    int resolution; 
    int BrickResolution;
};

#endif 