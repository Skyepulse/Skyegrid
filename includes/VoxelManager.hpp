#ifndef VOXELMANAGER_HPP
#define VOXELMANAGER_HPP

#include "../includes/Rendering/Pipelines/pipelines.hpp"
#include "../includes/constants.hpp"
#include "../includes/VoxelIO.hpp"
#include <cstdint>
#include <vector>
#include <array>
#include <map>
#include <atomic>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
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

// Async disk read limits
const int MAX_PENDING_DISK_READS = 256; // Max bricks queued for disk reading per frame
const int MAX_READY_BRICKS = 512;      // Max bricks ready to be uploaded per frame

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
    bool reading = false;
    bool pendingRead = false;
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
// Async disk read result
struct DiskReadResult
{
    uint32_t brickGridIndex;
    uint32_t occupancy[16];
    ColorRGB colors[512];
    bool success;
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

        startDiskReaderThread(); // This thread will be woken up and sleep as needed to read async bricks
    };
    ~VoxelManager()
    {
        stopDiskReaderThread();

        // free vectors
        brickGrid.clear();
    };

    void update(WgpuBundle& wgpuBundle, const wgpu::Queue& queue, const wgpu::CommandEncoder& encoder);
    void prepareFeedback(const wgpu::Queue& queue, const wgpu::CommandEncoder& encoder);
    void processAsyncOperations(wgpu::Instance& instance);

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
    int GetMaxVisibleBricks() const { return this->maxVisibleBricks; }
    void ChangeVoxelResolution(WgpuBundle& bundle, int newResolution, int maxVisibleBricks = -1)
    {
        int currentResolution = this->voxelResolution;
        int currentMaxVisibleBricks = this->maxVisibleBricks;
        if (maxVisibleBricks < 0)
            maxVisibleBricks = currentMaxVisibleBricks;
        validateResolution(bundle, newResolution, maxVisibleBricks);
        if (currentResolution == this->voxelResolution && currentMaxVisibleBricks == this->maxVisibleBricks)
            return; // We literally did not change anything
            
        // Clear any pending feedback/uploads that reference old brick indices
        feedbackRequests.clear();
        dirtyBrickIndices.clear();
        hasPendingFeedback = false;
        pendingUploadCount = 0;

        clearDiskReadQueues();
        
        initDynamicBuffers(bundle);
    }

    void loadFile(const std::string& filename);

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

    wgpu::Buffer brickRequestFlagsBuffer;
    wgpu::Buffer brickRequestFlagsRESET;

    // pools
    std::array<UploadBufferSlot, NUM_UPLOAD_BUFFERS> uploadBufferSlots;
    int currentUploadSlot = 0;
    std::array<FeedbackBufferSlot, NUM_FEEDBACK_BUFFERS> feedbackBufferSlots;
    int currentFeedbackWriteSlot = 0;  // Slot GPU writes to
    int currentFeedbackReadSlot = 0;   // Slot CPU reads from

    uint32_t pendingUploadCount = 0;
    uint32_t numberOfColorPools = 0;
    uint32_t maxColorBufferEntries = 0;

    uint64_t lastBrickIndex = 0; // DEBUG
    
    bool hasPendingFeedback = false;

    std::vector<uint32_t> feedbackRequests;
    std::vector<uint32_t> freeBrickSlots;
    std::vector<uint32_t> dirtyBrickIndices;

private:

    void validateResolution(WgpuBundle& bundle, int resolution, int maxVisibleBricks);
    ColorRGB computeBrickAverageColor(const BrickMapCPU& brick);
    void cleanupBuffers();

    void requestUploadBufferMap(int slotIndex);
    void requestFeedbackBufferMap(int slotIndex);
    void processPendingFeedback();
    void requestRead(const std::vector<uint32_t>& indices);

    // Async disk reading thread methods
    void startDiskReaderThread();
    void stopDiskReaderThread();
    void diskReaderThreadFunc();
    void clearDiskReadQueues();
    void queueDiskRead(uint32_t brickGridIndex);
    void processCompletedDiskReads();

    int voxelResolution; 
    int BrickResolution;
    int maxVisibleBricks;

    bool hasColor = false;

    std::unique_ptr<VoxelFileReader> voxelFileReader;
    bool loadedMesh = false;

    // The thread
    std::thread diskReaderThread;
    std::atomic<bool> diskReaderThreadRunning = false;

    std::queue<uint32_t> diskReadRequestQueue; // So that we process them in order of arrival
    std::mutex diskReadQueueMutex;
    std::condition_variable diskReadQueueCV;

    std::queue<DiskReadResult> diskReadResultQueue;
    std::mutex diskReadResultMutex;

    std::mutex fileReadMutex; // To protect file reading operations during async reads
};

#endif 