#ifndef VOXELMANAGER_HPP
#define VOXELMANAGER_HPP

#include "../includes/Rendering/Pipelines/pipelines.hpp"
#include "../includes/constants.hpp"
#include <cstdint>
#include <vector>
#include <array>
#include <map>
#include <atomic>
#include <mutex>
#include <iostream>

class VoxelManager; // Forward declaration

const int MAX_FEEDBACK = 8192;

// Max bricks is max index that we can pack in 24 bits, which is 2^24 - 1
const int MAX_BRICKS = 16777215;
const int COLOR_BYTES_PER_BRICK = 2048; // 8x8x8 voxels, 1 byte per voxel (RGB packed), aligned to 2048 bytes
const int MAX_COLOR_POOLS = 3;

// Number of buffered frames for async operations
const int NUM_UPLOAD_BUFFERS = 2;
const int NUM_FEEDBACK_BUFFERS = 2;

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
};

struct BrickGridCellCPU
{
    bool dirty = false;
    bool onGPU = false;
    uint32_t gpuBrickIndex = UINT32_MAX;
    ColorRGB LODColor;
};

struct BrickGridCell
{
    // [23:0]   : pointer / index or LOD in case unl oaded (r, g, b)
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
    uint32_t hasColor;
    uint32_t _pad;
};

//================================//
// Buffer states for async work
enum class BufferState
{
    Available,
    MappingInFlight,
    Mapped
};

struct UploadBufferSlot
{
    wgpu::Buffer cpuBuffer;      // MapWrite | CopySrc
    BufferState state = BufferState::Available;
    uint32_t pendingCount = 0;
};

struct FeedbackBufferSlot
{
    wgpu::Buffer cpuBuffer;      // MapRead | CopyDst
    BufferState state = BufferState::Available;
};

//================================//
class VoxelManager
{
public:
    VoxelManager(WgpuBundle& bundle, int resolution, int maxVisibleBricks)
    {
        static_assert(sizeof(ColorRGB) == 4); // packed in UINT32
        this->hasColor = HAS_VOXEL_COLOR;
        validateResolution(bundle, resolution, maxVisibleBricks);
    };
    ~VoxelManager()
    {
        // free vectors
        brickGrid.clear();
    };

    void update(WgpuBundle& wgpuBundle, const wgpu::Queue& queue, const wgpu::CommandEncoder& encoder);
    void prepareFeedback(const wgpu::Queue& queue, const wgpu::CommandEncoder& encoder);
    void processAsyncOperations(wgpu::Instance& instance);

    void setVoxel(int x, int y, int z, bool filled, ColorRGB color);
    void initDynamicBuffers(WgpuBundle& wgpuBundle);
    void initStaticBuffers(WgpuBundle& wgpuBundle);
    void createUploadBindGroup(RenderPipelineWrapper& pipelineWrapper, WgpuBundle& wgpuBundle);
    void startOfFrame();

    inline uint32_t BrickGridIndex(uint32_t bx, uint32_t by, uint32_t bz)
    {
        return bx + by * BrickResolution + bz * BrickResolution * BrickResolution;
    }

    bool GetHasColor() const { return this->hasColor; }
    int GetVoxelResolution() const { return this->voxelResolution; }
    void ChangeVoxelResolution(WgpuBundle& bundle, int newResolution)
    {
        int currentResolution = this->voxelResolution;
        validateResolution(bundle, newResolution, this->maxVisibleBricks);
        if (currentResolution == this->voxelResolution)
            return; // We literally did not change anything
            
        // Clear any pending feedback/uploads that reference old brick indices
        feedbackRequests.clear();
        dirtyBrickIndices.clear();
        hasPendingFeedback = false;
        pendingUploadCount = 0;
        
        initDynamicBuffers(bundle);
    }

    //CPU storage
    std::vector<BrickGridCell> brickGrid;
    std::vector<BrickGridCellCPU> brickGridCPU;
    std::map<uint32_t, BrickMapCPU> brickMaps;

    //GPU storage
    wgpu::Buffer brickGridBuffer;
    wgpu::Buffer brickPoolBuffer;

    std::vector<wgpu::Buffer> colorPoolBuffers;

    wgpu::Buffer feedbackCountBuffer;
    wgpu::Buffer feedbackCountRESET;
    wgpu::Buffer feedbackIndicesBuffer;

    wgpu::Buffer uploadBuffer;
    wgpu::Buffer uploadCountUniform;

    // pools
    std::array<UploadBufferSlot, NUM_UPLOAD_BUFFERS> uploadBufferSlots;
    int currentUploadSlot = 0;
    std::array<FeedbackBufferSlot, NUM_FEEDBACK_BUFFERS> feedbackBufferSlots;
    int currentFeedbackWriteSlot = 0;  // Slot GPU writes to
    int currentFeedbackReadSlot = 0;   // Slot CPU reads from

    std::vector<uint32_t> feedbackRequests;
    std::vector<uint32_t> freeBrickSlots;
    std::vector<uint32_t> dirtyBrickIndices;

    uint32_t pendingUploadCount = 0;
    uint32_t numberOfColorPools = 0;
    uint32_t maxColorBufferEntries = 0;

    uint64_t lastBrickIndex = 0; // DEBUG
    
    bool hasPendingFeedback = false;

private:

    void validateResolution(WgpuBundle& bundle, int resolution, int maxVisibleBricks);
    ColorRGB computeBrickAverageColor(const BrickMapCPU& brick);
    void cleanupBuffers();

    void requestUploadBufferMap(int slotIndex);
    void requestFeedbackBufferMap(int slotIndex);
    void processPendingFeedback();

    int voxelResolution; 
    int BrickResolution;
    int maxVisibleBricks;

    bool hasColor = false;
};

#endif 