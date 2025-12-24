#include "../includes/VoxelManager.hpp"
#include <iostream>
#include <bitset>

//================================//
struct feedbackCallbackContext 
{
    wgpu::Buffer mapBuffer;
    size_t bufferSize;
    std::vector<uint32_t>* feedbackIndices;
    bool* done;
};

//================================//
static uint32_t PackLOD(ColorRGB c)
{
    // pack the LOD in [23:0] as r (8 bits), g (8 bits), b (8 bits), and then resident [31] = 0, requested [30] = 0, unloaded [29] = 1
    return  (uint32_t(c.r)      |
            (uint32_t(c.g) << 8)  |
            (uint32_t(c.b) << 16) |
            (1u << 29));              // Set unloaded flag
}

//================================//
static uint32_t PackResident(uint32_t index)
{
    // pack the resident index in [23:0] which is guaranteed to be under 24 bits
    return  (index & 0x00FFFFFFu) | (1u << 31); // Set resident flag
}

//================================//
static uint32_t PackEmptyPointer()
{
    return 0u;
}

//================================//
ColorRGB VoxelManager::computeBrickAverageColor(const BrickMapCPU& brick)
{
    uint64_t count = 0;
    uint64_t r = 0, g = 0, b = 0;

    for (int z = 0; z < 8; ++z)
    {
        uint32_t firstSliceHalf = brick.occupancy[2* z];
        uint32_t secondSliceHalf = brick.occupancy[2* z + 1];
        uint64_t slice = (static_cast<uint64_t>(secondSliceHalf) << 32) | static_cast<uint64_t>(firstSliceHalf);
        while (slice)
        {
            int idx = 0;
            uint64_t tmp = slice;
            while ((tmp & 1ull) == 0ull)
            {
                tmp >>= 1;
                ++idx;
            }

            const ColorRGB& c = brick.colors[z * 64 + idx]; // Slices of 64, 64 * 8 = 512 voxels per brick
            r += c.r;
            g += c.g;
            b += c.b;
            ++count;

            // clear lowest set bit
            slice &= slice - 1; // Clear the lowest set bit
        }
    }

    if (count == 0)
        return {0,0,0};

    return {
        uint8_t(r / count),
        uint8_t(g / count),
        uint8_t(b / count)
    };
}

//================================//
void VoxelManager::setVoxel(int x, int y, int z, bool filled, ColorRGB color)
{
    int bx = x / 8;
    int by = y / 8;
    int bz = z / 8;

    int brickIndex = bx + by * this->BrickResolution + bz * this->BrickResolution * this->BrickResolution;
    BrickMapCPU& brick = brickMaps[brickIndex];

    int lx = x % 8;
    int ly = y % 8;
    int lz = z % 8;
    int bit = lx + ly * 8;

    int sliceBase = lz * 2;

    if (filled)
    {
        // occupancy is now 16 half slices of 32 bits
        if (bit < 32) {
            brick.occupancy[sliceBase] |= (1u << bit);
        } else {
            brick.occupancy[sliceBase + 1] |= (1u << (bit - 32));
        }
        brick.colors[lz * 64 + bit] = color;
    }
    else
    {
        // todo clear correct bit in first or second half
        if (bit < 32) {
            brick.occupancy[sliceBase] &= ~(1u << bit);
        } else {
            brick.occupancy[sliceBase + 1] &= ~(1u << (bit - 32));
        }
        brick.colors[lz * 64 + bit] = {0,0,0};
    }

    brick.lodColor = computeBrickAverageColor(brick); // Recompute LOD color average
    if (brick.dirty == false)
        dirtyBrickIndices.push_back(brickIndex); // Do not push if already dirty
    brick.dirty = true;
}

//================================//
void VoxelManager::startOfFrame()
{
    // At the start of the frame, reset dirty brick indices
    dirtyBrickIndices.clear();
    pendingUploadCount = 0;

    // Read all feedback requests, and for each requested brick, load the first voxel
    for (uint32_t requestedBrickIndex : feedbackRequests)
    {
        BrickMapCPU& brick = brickMaps[requestedBrickIndex];

        // For testing, set the first voxel in the brick to red
        // Compute voxel coordinates
        int bx = requestedBrickIndex % BrickResolution;
        int by = (requestedBrickIndex / BrickResolution) % BrickResolution;
        int bz = requestedBrickIndex / (BrickResolution * BrickResolution);

        int vx = bx * 8;
        int vy = by * 8;
        int vz = bz * 8;

        this->setVoxel(vx, vy, vz, true, ColorRGB{ 255, 0, 0 }); // set first voxel to red
    }
}

//================================//
// called when unmapping a buffer to map it again, with some luck it will be ready by next frame
// if not we create another buffer and make a pool out of them
bool VoxelManager::remapUploadBuffer(wgpu::Future& outFuture)
{
    if (!requestRemap)
    {
        return false;
    }

    static auto OnMapped = [](wgpu::MapAsyncStatus status,
                            wgpu::StringView message,
                            VoxelManager* userdata)
    {
        if (status != wgpu::MapAsyncStatus::Success)
        {
            std::cerr << "[VoxelManager] Failed to remap upload buffer: " << std::string(message) << std::endl;
            return;
        }
    };

    // Start the async mapping operation
    outFuture = CPUuploadBuffer.MapAsync(
        wgpu::MapMode::Write,
        0,
        MAX_FEEDBACK * sizeof(UploadEntry),
        wgpu::CallbackMode::WaitAnyOnly,
        OnMapped,
        this
    );

    requestRemap = false;
    return true;
}

//================================//
void VoxelManager::update(WgpuBundle& wgpuBundle, const wgpu::Queue& queue, const wgpu::CommandEncoder& encoder)
{
    if (dirtyBrickIndices.empty())
        return;

    requestRemap = true;

    // Map the CPU upload buffer (MapWrite | CopySrc)
    UploadEntry* uploads = static_cast<UploadEntry*>(
        CPUuploadBuffer.GetMappedRange(0, MAX_FEEDBACK * sizeof(UploadEntry))
    );

    for (uint32_t brickGridIndex : dirtyBrickIndices)
    {
        BrickMapCPU& brick = brickMaps[brickGridIndex];

        // if it is dirty and not on gpu, we must allocate a brick slot
        if (!brick.onGPU)
        {
            if (freeBrickSlots.empty())
            {
                // No free brick slots available on GPU
                continue;
            }

            // Allocate a brick slot
            brick.gpuBrickIndex = freeBrickSlots.back();
            freeBrickSlots.pop_back();
            brick.onGPU = true;
        }

        if (pendingUploadCount >= MAX_FEEDBACK) break;

        UploadEntry& entry = uploads[pendingUploadCount++];
        entry.gpuBrickSlot = brick.gpuBrickIndex; // The free slot allocated above
        std::memcpy(entry.occupancy, brick.occupancy, sizeof(entry.occupancy));
        std::memcpy(entry.colors, brick.colors, sizeof(entry.colors));

        brickGrid[brickGridIndex].pointer = PackResident(brick.gpuBrickIndex);
        brick.dirty = false;
    }
    CPUuploadBuffer.Unmap();

    if (pendingUploadCount > 0)
    {
        encoder.CopyBufferToBuffer(
            CPUuploadBuffer, 0,               // Source: CPU upload buffer (MapWrite | CopySrc)
            uploadBuffer, 0,                  // Destination: GPU upload buffer (Storage | CopyDst)
            pendingUploadCount * sizeof(UploadEntry)
        );
    }

    // Set feedback count to 0 for next frame
    encoder.CopyBufferToBuffer(
        feedbackCountRESET, 0,            // Source: reset buffer (MapWrite | CopySrc)
        feedbackCountBuffer, 0,              // Destination: GPU buffer (Storage | CopyDst)
        sizeof(uint32_t)
    );

    // Write updated brick grid to GPU (no mapping, just write)
    queue.WriteBuffer(
        brickGridBuffer,
        0,
        brickGrid.data(),
        brickGrid.size() * sizeof(BrickGridCell)
    );
}

//================================//
void VoxelManager::prepareFeedback(const wgpu::Queue& queue, const wgpu::CommandEncoder& encoder)
{
    // read the feedback buffer from GPU and process requested bricks
    encoder.CopyBufferToBuffer(
        feedbackCountBuffer, 0,     
        CPUfeedbackBuffer, 0, 
        sizeof(uint32_t)
    );

    // We wrote a uint32_t count followed by MAX_FEEDBACK uint32_t indices
    encoder.CopyBufferToBuffer(
        feedbackIndicesBuffer, 0,     
        CPUfeedbackBuffer, sizeof(uint32_t), 
        MAX_FEEDBACK * sizeof(uint32_t)
    );
}

//================================//
void VoxelManager::readFeedback(wgpu::Future& outFuture)
{
    // Now map the CPU feedback count buffer to read
    static auto OnMapped = [](  wgpu::MapAsyncStatus status,
                                wgpu::StringView message,
                                feedbackCallbackContext* userdata)
    {
        auto* ctx = reinterpret_cast<feedbackCallbackContext*>(userdata);

        if (status == wgpu::MapAsyncStatus::Success)
        {
            // When reading, the first uint32_t is the count, and then follow the indices
            const uint8_t* mappedData = static_cast<const uint8_t*>(ctx->mapBuffer.GetConstMappedRange(0, ctx->bufferSize));
            uint32_t feedbackCount = *reinterpret_cast<const uint32_t*>(mappedData);
            if (feedbackCount > MAX_FEEDBACK)
                feedbackCount = MAX_FEEDBACK;
            else if (feedbackCount == 0)
            {
                ctx->mapBuffer.Unmap();
                *(ctx->done) = true;
                delete ctx;
                return;
            }

            ctx->feedbackIndices->resize(feedbackCount);
            const uint32_t* indicesPtr = reinterpret_cast<const uint32_t*>(mappedData + sizeof(uint32_t));
            for (uint32_t i = 0; i < feedbackCount; ++i)
            {
                (*ctx->feedbackIndices)[i] = indicesPtr[i];
            }

            ctx->mapBuffer.Unmap();
        }
        else
        {
            std::cerr << "Failed to map buffer: " << message << std::endl;
        }

        *(ctx->done) = true;
        delete ctx;
    };

    feedbackCallbackContext* context = new feedbackCallbackContext();
    context->mapBuffer = CPUfeedbackBuffer;
    context->bufferSize = sizeof(uint32_t) + MAX_FEEDBACK * sizeof(uint32_t);
    context->done = new bool(false);
    context->feedbackIndices = &this->feedbackRequests; 
    outFuture = CPUfeedbackBuffer.MapAsync(
        wgpu::MapMode::Read,
        0,
        context->bufferSize,
        wgpu::CallbackMode::WaitAnyOnly,
        OnMapped,
        context
    );
}

//================================//
void VoxelManager::initBuffers(WgpuBundle& wgpuBundle)
{
    std::cout << "[VoxelManager] Initializing buffers on CPU side...\n";
    // CPU storage initialization
    this->brickGrid.resize(this->BrickResolution * this->BrickResolution * this->BrickResolution);
    for(int i = 0; i < this->brickGrid.size(); ++i)
    {
        this->brickGrid[i].pointer = PackLOD({255, 255, 0}); // Yellow: 255 red, 255 green, 0 blue LOD for unloaded bricks
    }

    const uint32_t MAX_GPU_BRICKS = this->BrickResolution * this->BrickResolution * this->BrickResolution;
    this->freeBrickSlots.reserve(MAX_GPU_BRICKS);
    for (uint32_t i = 0; i < MAX_GPU_BRICKS; ++i)
    {
        freeBrickSlots.push_back(i);
    }

    this->brickMaps.resize(this->BrickResolution * this->BrickResolution * this->BrickResolution);
    for (BrickMapCPU& brick : this->brickMaps)
    {
        std::memset(brick.occupancy, 0, sizeof(brick.occupancy)); // Init to empty
        std::memset(brick.colors, 0, sizeof(brick.colors)); // Init to black
        brick.lodColor = {0,0,0};
        brick.dirty = false;
        brick.onGPU = false;
        brick.gpuBrickIndex = UINT32_MAX;
    }

    std::cout << "[VoxelManager] Initializing buffers on GPU side...\n";
    // GPU storage initialization
    wgpu::Queue queue = wgpuBundle.GetDevice().GetQueue();

    // [1] BRICK GRID CELL
    wgpu::BufferDescriptor desc{};
    desc.size = brickGrid.size() * sizeof(BrickGridCell); // a single uint64_t pointer per cell
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    desc.label = "Brick Grid Buffer";
    desc.mappedAtCreation = false;
    this->brickGridBuffer = wgpuBundle.GetDevice().CreateBuffer(&desc);
    queue.WriteBuffer(brickGridBuffer, 0, brickGrid.data(), desc.size);

    // [2] BRICK POOL BUFFER
    // The max number of bricks, is given by the Maximum Voxel Resolution divided by 8 (brick size)
    desc.size = MAX_GPU_BRICKS * 16 * sizeof(uint32_t); // 8x8x8 occupancy per brick, times MAX_GPU_BRICKS
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    desc.label = "Brick Pool Buffer";
    desc.mappedAtCreation = false;
    this->brickPoolBuffer = wgpuBundle.GetDevice().CreateBuffer(&desc);

    // [3] COLOR POOL BUFFER
    // 512 voxels * 3 bytes per voxel
    const uint32_t colorPerBrickSize = 2048; // for alignment
    desc.size = MAX_GPU_BRICKS * colorPerBrickSize;
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    desc.label = "Color Pool Buffer";
    desc.mappedAtCreation = false;
    this->colorPoolBuffer = wgpuBundle.GetDevice().CreateBuffer(&desc);

    // [4] FEEDBACK BUFFERS
    // feedbackCount init is a single uint32_t set to 0
    uint32_t feedbackCountInit = 0;
    desc.size = sizeof(uint32_t);
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
    desc.label = "Feedback Count Buffer (GPU)";
    desc.mappedAtCreation = false;
    this->feedbackCountBuffer = wgpuBundle.GetDevice().CreateBuffer(&desc);

    // Reset buffer for feedback count
    desc.size = sizeof(uint32_t);
    desc.usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc;
    desc.mappedAtCreation = true; // So we can initialize it to 0 right away
    desc.label = "Feedback Count Reset Buffer (RESET)";
    this->feedbackCountRESET = wgpuBundle.GetDevice().CreateBuffer(&desc);
    memcpy(feedbackCountRESET.GetMappedRange(), &feedbackCountInit, sizeof(uint32_t));
    feedbackCountRESET.Unmap();

    // now the feedback indices buffer
    desc.size = MAX_FEEDBACK * sizeof(uint32_t);
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    desc.label = "Feedback Indices Buffer (GPU)";
    desc.mappedAtCreation = false;
    this->feedbackIndicesBuffer = wgpuBundle.GetDevice().CreateBuffer(&desc);

    // CPU feedback indices AND count buffer
    desc.size = sizeof(uint32_t) + MAX_FEEDBACK * sizeof(uint32_t);
    desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    desc.label = "Feedback Buffer (indices AND count) (CPU)";
    desc.mappedAtCreation = false;
    this->CPUfeedbackBuffer = wgpuBundle.GetDevice().CreateBuffer(&desc);

    // Upload buffer initialization
    desc.size = MAX_FEEDBACK * sizeof(UploadEntry);
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    desc.label = "Upload Buffer (GPU)";
    desc.mappedAtCreation = false;
    this->uploadBuffer = wgpuBundle.GetDevice().CreateBuffer(&desc);

    // CPU upload buffer (one, then maybe we create more)
    desc.size = MAX_FEEDBACK * sizeof(UploadEntry);
    desc.usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc;
    desc.label = "Upload Buffer (CPU)";
    desc.mappedAtCreation = true; // So we can write to it right away
    this->CPUuploadBuffer = wgpuBundle.GetDevice().CreateBuffer(&desc);

    // Upload count uniform buffer
    desc.size = sizeof(uint32_t);
    desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    desc.label = "Upload Count Uniform Buffer";
    desc.mappedAtCreation = false;
    this->uploadCountUniform = wgpuBundle.GetDevice().CreateBuffer(&desc);
}

//================================//
void VoxelManager::createUploadBindGroup(RenderPipelineWrapper& pipelineWrapper, WgpuBundle& wgpuBundle)
{
    wgpu::BindGroupEntry entries[4]{};

    entries[0].binding = 0;
    entries[0].buffer = this->uploadBuffer;
    entries[0].offset = 0;
    entries[0].size = this->uploadBuffer.GetSize();

    entries[1].binding = 1;
    entries[1].buffer = this->brickPoolBuffer;
    entries[1].offset = 0;
    entries[1].size = this->brickPoolBuffer.GetSize();

    entries[2].binding = 2;
    entries[2].buffer = this->colorPoolBuffer;
    entries[2].offset = 0;
    entries[2].size = this->colorPoolBuffer.GetSize();

    entries[3].binding = 3;
    entries[3].buffer = this->uploadCountUniform;
    entries[3].offset = 0;
    entries[3].size = this->uploadCountUniform.GetSize();

    wgpu::BindGroupDescriptor bindGroupDesc{};
    bindGroupDesc.layout = pipelineWrapper.bindGroupLayout;
    bindGroupDesc.entryCount = 4;
    bindGroupDesc.entries = entries;

    pipelineWrapper.bindGroup = wgpuBundle.GetDevice().CreateBindGroup(&bindGroupDesc);
}