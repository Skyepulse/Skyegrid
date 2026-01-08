#include "../includes/VoxelManager.hpp"
#include "../includes/constants.hpp"
#include <iostream>
#include <bitset>
#include <string>
#include <fstream>

//================================//
struct UploadMapCallbackContext 
{
    VoxelManager* voxelManager;
    int slotIndex;
};

struct FeedbackMapCallbackContext 
{
    VoxelManager* voxelManager;
    int slotIndex;
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
void VoxelManager::validateResolution(WgpuBundle& bundle, int resolution, int maxVisibleBricks)
{
    // First of all, calculate limits to see if we support this resolution
    uint64_t maxBufferSize = bundle.GetLimits().maxBufferSize;
    uint64_t maxColorBufferSize = (maxBufferSize / static_cast<uint64_t>(COLOR_BYTES_PER_BRICK)) * static_cast<uint64_t>(COLOR_BYTES_PER_BRICK);
    this->maxColorBufferEntries = static_cast<uint32_t>(maxColorBufferSize / sizeof(ColorRGB)); // in number of ColorRGB entries per buffer pool entry

    if (resolution <= 0)
    {
        std::cout << "[VoxelManager] Voxel resolution must be positive." << std::endl;
        throw std::runtime_error("[VoxelManager] Voxel resolution must be positive.");
    }
    else if (resolution < 8)
    {
        this->voxelResolution = 8;
        this->BrickResolution = 1;
        this->numberOfColorPools = this->hasColor ? 1 : 0;
        this->maxVisibleBricks = 1;

        std::cout << "[VoxelManager] Voxel resolution too low, clamping to 8." << std::endl;
        return;
    }

    if (this->hasColor)
    {
        // Now compute maximum possible resolution of visible bricks
        uint64_t maxTotalVisibleColorSize = uint64_t(MAX_COLOR_POOLS) * maxColorBufferSize;
        uint64_t maxVisibleBricksPossible = maxTotalVisibleColorSize / COLOR_BYTES_PER_BRICK;

        std::cout << "[VoxelManager] Max visible bricks possible with current device limits: " << maxVisibleBricksPossible << std::endl;
        std::cout << "We wanted to have... " << maxVisibleBricks << " visible bricks." << std::endl;

        // [1] clamp the max visible bricks to the possible maximum, with our 3 color pools
        maxVisibleBricks = std::min(static_cast<uint64_t>(maxVisibleBricks), maxVisibleBricksPossible - 1);
    }
    else    
    {
        this->numberOfColorPools = 0;
    }

    if (resolution % 8 != 0)
    {
        resolution = (resolution / 8) * 8; // floor to nearest multiple of 8
    }

    // [2] check if the resolution forces a number of bricks higher than 24 bit encoding, max possible for our brick grid
    int brickResolution = resolution / 8;
    uint64_t numBricks = static_cast<uint64_t>(brickResolution) * static_cast<uint64_t>(brickResolution) * static_cast<uint64_t>(brickResolution);
    if (numBricks > MAX_BRICKS)
    {
        // reduce resolution until it fits
        while (numBricks >= MAX_BRICKS)
        {
            resolution -= 8;
            brickResolution = resolution / 8;
            numBricks = static_cast<uint64_t>(brickResolution) * static_cast<uint64_t>(brickResolution) * static_cast<uint64_t>(brickResolution);
        }
    }

    this->voxelResolution = resolution;
    this->BrickResolution = resolution / 8;

    // Now clamp max visible bricks to total number of bricks, we cannot have more visible bricks than total bricks
    maxVisibleBricks = std::min(maxVisibleBricks, static_cast<int>(numBricks));
    this->maxVisibleBricks = maxVisibleBricks;
    
    // Compute number of color pools needed
    if (this->hasColor)
    {
        uint64_t totalColorBytesNeeded = static_cast<uint64_t>(this->maxVisibleBricks) * static_cast<uint64_t>(COLOR_BYTES_PER_BRICK);
        this->numberOfColorPools = static_cast<uint32_t>((totalColorBytesNeeded + maxColorBufferSize - 1) / maxColorBufferSize);
        if (this->numberOfColorPools > MAX_COLOR_POOLS)
        {
            std::cout << "[VoxelManager] Unable to allocate enough color pool buffers for the requested visible bricks (" << maxVisibleBricks << "). Max supported visible bricks is lower. THIS SHOULD NOT HAPPEN." << std::endl;
            throw std::runtime_error("[VoxelManager] Unable to allocate enough color pool buffers for the requested visible bricks.");
        }
    }

    std::cout << "[VoxelManager] Voxel resolution set to " << this->voxelResolution << " (" << static_cast<uint64_t>(this->voxelResolution) * this->voxelResolution * this->voxelResolution << " total voxels)." << std::endl;
    std::cout << "[VoxelManager] Max visible bricks set to " << maxVisibleBricks << "." << std::endl;
    std::cout << "[VoxelManager] Using " << numberOfColorPools << " color pool buffers." << std::endl;
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
void VoxelManager::startOfFrame()
{
    // At the start of the frame, reset dirty brick indices
    dirtyBrickIndices.clear();
    pendingUploadCount = 0;

    // Process any pending feedback that was read from previous frames
    processPendingFeedback();
}

//================================//
void VoxelManager::processPendingFeedback()
{
    if (!hasPendingFeedback)
        return;

    const uint32_t maxValidIndex = static_cast<uint32_t>(brickGridCPU.size());
    for (uint32_t requestedBrickIndex : this->feedbackRequests)
    {
        if (requestedBrickIndex < maxValidIndex)
        {
            this->dirtyBrickIndices.push_back(requestedBrickIndex);
        }
    }

    this->feedbackRequests.clear();
    this->hasPendingFeedback = false;
}

//================================//
void VoxelManager::requestUploadBufferMap(int slotIndex)
{
    UploadBufferSlot& slot = uploadBufferSlots[slotIndex];
    
    if (slot.state != BufferState::Available)
        return; // Already mapping or mapped, we cannot request again on this slot
    
    slot.state = BufferState::MappingInFlight;
    
    UploadMapCallbackContext* ctx = new UploadMapCallbackContext{this, slotIndex};
    
    slot.cpuBuffer.MapAsync(
        wgpu::MapMode::Write,
        0,
        MAX_FEEDBACK * sizeof(UploadEntry),
        wgpu::CallbackMode::AllowProcessEvents,
        [](wgpu::MapAsyncStatus status, wgpu::StringView message, UploadMapCallbackContext* ctx) {
            if (status == wgpu::MapAsyncStatus::Success)
            {
                ctx->voxelManager->uploadBufferSlots[ctx->slotIndex].state = BufferState::Mapped;
            }
            else
            {
                ctx->voxelManager->uploadBufferSlots[ctx->slotIndex].state = BufferState::Available;
                std::cerr << "[VoxelManager] Upload buffer map failed for slot " << ctx->slotIndex << std::endl;
            }
            delete ctx;
        },
        ctx
    );
}

//================================//
void VoxelManager::requestFeedbackBufferMap(int slotIndex)
{
    FeedbackBufferSlot& slot = feedbackBufferSlots[slotIndex];
    
    if (slot.state != BufferState::Available)
        return; // Same dhere
    
    slot.state = BufferState::MappingInFlight;
    
    // Create context for callback
    FeedbackMapCallbackContext* ctx = new FeedbackMapCallbackContext{this, slotIndex};
    
    size_t bufferSize = sizeof(uint32_t) + MAX_FEEDBACK * sizeof(uint32_t);
    
    slot.cpuBuffer.MapAsync(
        wgpu::MapMode::Read,
        0,
        bufferSize,
        wgpu::CallbackMode::AllowProcessEvents,
        [](wgpu::MapAsyncStatus status, wgpu::StringView message, FeedbackMapCallbackContext* ctx) {
            FeedbackBufferSlot& slot = ctx->voxelManager->feedbackBufferSlots[ctx->slotIndex];
            
            if (status == wgpu::MapAsyncStatus::Success)
            {
                // Read the feedback data immediately while mapped
                const uint8_t* mappedData = static_cast<const uint8_t*>(
                    slot.cpuBuffer.GetConstMappedRange()
                );
                
                uint32_t count = *reinterpret_cast<const uint32_t*>(mappedData);
                count = std::min(count, static_cast<uint32_t>(MAX_FEEDBACK));
                
                const uint32_t* indices = reinterpret_cast<const uint32_t*>(
                    mappedData + sizeof(uint32_t)
                );
                
                // Copy feedback data
                ctx->voxelManager->feedbackRequests.resize(count);
                if (count > 0)
                {
                    memcpy(ctx->voxelManager->feedbackRequests.data(), indices, count * sizeof(uint32_t));
                }
                
                ctx->voxelManager->hasPendingFeedback = true;
                
                // Unmap immediately after reading
                slot.cpuBuffer.Unmap();
                slot.state = BufferState::Available;
            }
            else
            {
                slot.state = BufferState::Available;
                std::cerr << "[VoxelManager] Feedback buffer map failed for slot " << ctx->slotIndex << std::endl;
            }
            
            delete ctx;
        },
        ctx
    );
}

//================================//
// MAIN POTIN FOR OUR ASYNC PROCESSING
void VoxelManager::processAsyncOperations(wgpu::Instance& instance)
{
    instance.ProcessEvents(); // Apparently this fires the callbacks for mapping
    
    for (int i = 0; i < NUM_UPLOAD_BUFFERS; ++i)
    {
        if (uploadBufferSlots[i].state == BufferState::Available)
        {
            requestUploadBufferMap(i);
        }
    }
}

//================================//
void VoxelManager::update(WgpuBundle& wgpuBundle, const wgpu::Queue& queue, const wgpu::CommandEncoder& encoder)
{
    // Try to find an available mapped upload buffer
    int availableSlot = -1;
    for (int i = 0; i < NUM_UPLOAD_BUFFERS; ++i)
    {
        if (uploadBufferSlots[i].state == BufferState::Mapped)
        {
            availableSlot = i;
            break;
        }
    }

    // if we find NO available mapped buffer, therefore we cannot upload this frame, 
    // or no need to upload anything, szwe pass
    if (availableSlot < 0 || dirtyBrickIndices.empty())
    {
        // Still reset feedback count for next frame (htis is free)
        encoder.CopyBufferToBuffer(
            feedbackCountRESET, 0,
            feedbackCountBuffer, 0,
            sizeof(uint32_t)
        );
        
        encoder.CopyBufferToBuffer(
            brickRequestFlagsRESET, 0,
            brickRequestFlagsBuffer, 0,
            brickRequestFlagsBuffer.GetSize()
        );

        return;
    }

    // If we are here, we can map and upload
    UploadBufferSlot& slot = uploadBufferSlots[availableSlot];
    
    // Get the mapped range
    UploadEntry* uploads = static_cast<UploadEntry*>(slot.cpuBuffer.GetMappedRange(0, MAX_FEEDBACK * sizeof(UploadEntry)));
    if (!uploads)
    {
        std::cerr << "[VoxelManager] Failed to get mapped range for upload buffer" << std::endl;
        return;
    }

    std::vector<uint32_t> modifiedIndices;
    modifiedIndices.reserve(dirtyBrickIndices.size());

    for (uint32_t brickGridIndex : dirtyBrickIndices)
    {
        BrickGridCellCPU& brick = brickGridCPU[brickGridIndex];

        // if it is dirty and not on gpu, we must allocate a brick slot
        if (!brick.onGPU)
        {
            if (freeBrickSlots.empty())
            {
                continue;
            }

            brick.gpuBrickIndex = freeBrickSlots.back();
            freeBrickSlots.pop_back();
            brick.onGPU = true;

            BrickMapCPU& brickMap = brickMaps[brickGridIndex] = {};
            brickMap.occupancy[0] = 1u;
            for (int i = 1; i < 16; ++i)
                brickMap.occupancy[i] = 0u;
            brickMap.colors[0] = {
                static_cast<uint8_t>(rand() % 256),
                static_cast<uint8_t>(rand() % 256),
                static_cast<uint8_t>(rand() % 256)
            };
        }

        if (pendingUploadCount >= MAX_FEEDBACK) break;

        BrickMapCPU& brickMap = brickMaps[brickGridIndex];
        UploadEntry& entry = uploads[pendingUploadCount++];
        entry.gpuBrickSlot = brick.gpuBrickIndex;
        assert(entry.gpuBrickSlot < static_cast<uint32_t>(maxVisibleBricks));

        std::memcpy(entry.occupancy, brickMap.occupancy, sizeof(entry.occupancy));
        std::memcpy(entry.colors, brickMap.colors, sizeof(entry.colors));

        brickGrid[brickGridIndex].pointer = PackResident(brick.gpuBrickIndex);
        brick.dirty = false;
        modifiedIndices.push_back(brickGridIndex);
    }

    // Unmap the buffer before using it in a copy
    slot.cpuBuffer.Unmap();
    slot.state = BufferState::Available;  // Will be re-mapped by processAsyncOperations
    slot.pendingCount = pendingUploadCount;

    if (pendingUploadCount > 0)
    {
        encoder.CopyBufferToBuffer(
            slot.cpuBuffer, 0,
            uploadBuffer, 0,
            pendingUploadCount * sizeof(UploadEntry)
        );
    }

    // Set feedback count to 0 for next frame
    encoder.CopyBufferToBuffer(
        feedbackCountRESET, 0,
        feedbackCountBuffer, 0,
        sizeof(uint32_t)
    );

    encoder.CopyBufferToBuffer(
        brickRequestFlagsRESET, 0,
        brickRequestFlagsBuffer, 0,
        brickRequestFlagsBuffer.GetSize()
    );

    // Write updated brick grid to GPU
    if (!modifiedIndices.empty())
    {
        std::sort(modifiedIndices.begin(), modifiedIndices.end());
        
        const size_t count = modifiedIndices.size();
        size_t i = 0;
        
        while (i < count)
        {
            uint32_t rangeStart = modifiedIndices[i];
            uint32_t rangeEnd = rangeStart;
            
            while (i + 1 < count && modifiedIndices[i + 1] == rangeEnd + 1)
            {
                ++i;
                ++rangeEnd;
            }
            
            // Write the contiguous range
            uint32_t rangeSize = rangeEnd - rangeStart + 1;
            queue.WriteBuffer(
                brickGridBuffer,
                rangeStart * sizeof(BrickGridCell),
                &brickGrid[rangeStart],
                rangeSize * sizeof(BrickGridCell)
            );
            i++;
        }
    }
}

//================================//
void VoxelManager::prepareFeedback(const wgpu::Queue& queue, const wgpu::CommandEncoder& encoder)
{
    // Find an available feedback buffer slot for writing
    int writeSlot = -1;
    for (int i = 0; i < NUM_FEEDBACK_BUFFERS; ++i)
    {
        // Only use buffers that are Available (not mapped or mapping)
        if (feedbackBufferSlots[i].state == BufferState::Available)
        {
            writeSlot = i;
            break;
        }
    }
    
    if (writeSlot < 0)
    {
        // No buffer available, all are being mapped/read
        // This is fine, we'll catch up next frame
        return;
    }
    
    currentFeedbackWriteSlot = writeSlot;
    
    // Copy feedback from GPU to the selected CPU buffer
    FeedbackBufferSlot& slot = feedbackBufferSlots[writeSlot];
    
    encoder.CopyBufferToBuffer(
        feedbackCountBuffer, 0,     
        slot.cpuBuffer, 0, 
        sizeof(uint32_t)
    );

    encoder.CopyBufferToBuffer(
        feedbackIndicesBuffer, 0,     
        slot.cpuBuffer, sizeof(uint32_t), 
        MAX_FEEDBACK * sizeof(uint32_t)
    );
    
    // After the encoder is submitted, we'll request a map on this buffer
    // We mark it for mapping after submit, we save which slot to map (where the information was copied to)
    currentFeedbackReadSlot = writeSlot;
}

//================================//
void VoxelManager::cleanupBuffers()
{
    this->brickGrid.clear();
    this->brickGridCPU.clear();
    this->brickMaps.clear();
    this->freeBrickSlots.clear();

    this->brickGridBuffer = nullptr;
    this->brickPoolBuffer = nullptr;
    this->colorPoolBuffers.clear();

    this->brickRequestFlagsBuffer = nullptr;
    this->brickRequestFlagsRESET = nullptr;
}

//================================//
void VoxelManager::initStaticBuffers(WgpuBundle& wgpuBundle)
{
    wgpu::BufferDescriptor desc{};

    // [4] FEEDBACK BUFFERS
    // feedbackCount init is a single uint32_t set to 0
    uint32_t feedbackCountInit = 0;
    desc.size = sizeof(uint32_t);
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
    desc.label = "Feedback Count Buffer (GPU)";
    desc.mappedAtCreation = false;
    wgpuBundle.SafeCreateBuffer(&desc, this->feedbackCountBuffer);

    // Reset buffer for feedback count
    desc.size = sizeof(uint32_t);
    desc.usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc;
    desc.mappedAtCreation = true; // So we can initialize it to 0 right away
    desc.label = "Feedback Count Reset Buffer (RESET)";
    wgpuBundle.SafeCreateBuffer(&desc, this->feedbackCountRESET);
    memcpy(feedbackCountRESET.GetMappedRange(), &feedbackCountInit, sizeof(uint32_t));
    feedbackCountRESET.Unmap();

    // now the feedback indices buffer
    desc.size = MAX_FEEDBACK * sizeof(uint32_t);
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    desc.label = "Feedback Indices Buffer (GPU)";
    desc.mappedAtCreation = false;
    wgpuBundle.SafeCreateBuffer(&desc, this->feedbackIndicesBuffer);

    // Double-buffered CPU feedback buffers
    for (int i = 0; i < NUM_FEEDBACK_BUFFERS; ++i)
    {
        desc.size = sizeof(uint32_t) + MAX_FEEDBACK * sizeof(uint32_t); // count (u32) + indices (u32[])
        desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
        desc.label = ("Feedback Buffer CPU " + std::to_string(i)).c_str();
        desc.mappedAtCreation = false;
        wgpuBundle.SafeCreateBuffer(&desc, feedbackBufferSlots[i].cpuBuffer);
        feedbackBufferSlots[i].state = BufferState::Available;
    }

    // Upload buffer initialization
    desc.size = MAX_FEEDBACK * sizeof(UploadEntry);
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    desc.label = "Upload Buffer (GPU)";
    desc.mappedAtCreation = false;
    wgpuBundle.SafeCreateBuffer(&desc, this->uploadBuffer);

    // Double-buffered CPU upload buffers
    for (int i = 0; i < NUM_UPLOAD_BUFFERS; ++i)
    {
        desc.size = MAX_FEEDBACK * sizeof(UploadEntry);
        desc.usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc;
        desc.label = ("Upload Buffer CPU " + std::to_string(i)).c_str();
        desc.mappedAtCreation = true; // SO ALL IN POOL ARE AVAILABLE INITIALLY
        wgpuBundle.SafeCreateBuffer(&desc, uploadBufferSlots[i].cpuBuffer);
        uploadBufferSlots[i].state = BufferState::Mapped;
        uploadBufferSlots[i].pendingCount = 0;
    }

    // Upload count uniform buffer
    desc.size = sizeof(UploadUniform);
    desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    desc.label = "Upload Count Uniform Buffer";
    desc.mappedAtCreation = false;
    wgpuBundle.SafeCreateBuffer(&desc, this->uploadCountUniform);
}

//================================//
void VoxelManager::initDynamicBuffers(WgpuBundle& wgpuBundle)
{
    cleanupBuffers();

    const uint32_t numBricks = static_cast<uint64_t>(this->BrickResolution) * static_cast<uint64_t>(this->BrickResolution) * static_cast<uint64_t>(this->BrickResolution);
    const uint32_t numVisibleBricks = static_cast<uint32_t>(this->maxVisibleBricks);
    const uint64_t numVoxels = static_cast<uint64_t>(numBricks) * 512;

    // CPU storage initialization
    this->brickGrid.resize(numBricks);
    this->brickGridCPU.resize(numBricks);
    this->freeBrickSlots.resize(numVisibleBricks);

    // If loaded file, and matching resolution, get info on bricks here
    std::vector<brickIndexEntry> loadedBricks;
    bool matchingResolution = false;
    if (this->loadedMesh)
    {
        this->voxelFileReader->getInitialOccupiedBricks(loadedBricks);
        matchingResolution = (this->voxelFileReader->getResolution() == this->voxelResolution);
    }

    for(uint32_t i = 0; i < this->brickGrid.size(); ++i)
    {
        this->brickGrid[i].pointer = PackEmptyPointer();

        BrickGridCellCPU& cell = this->brickGridCPU[i];
        cell.dirty = false;
        cell.onGPU = false;
        cell.gpuBrickIndex = UINT32_MAX;
        cell.LODColor = {0,0,0};
    }
    for (uint32_t i = 0; i < numVisibleBricks; ++i)
    {
        freeBrickSlots[i] = numVisibleBricks - 1 - i;
    }

    if (matchingResolution)
    {
        for (const brickIndexEntry& entry : loadedBricks)
        {
            if (entry.brickGridIndex >= numBricks)
                continue;

            ColorRGB lod = {entry.LOD_R, entry.LOD_G, entry.LOD_B};
            this->brickGrid[entry.brickGridIndex].pointer = PackLOD(lod);

            BrickGridCellCPU& cell = this->brickGridCPU[entry.brickGridIndex];
            cell.LODColor = lod;
        }
    }

    // GPU storage initialization
    wgpu::Queue queue = wgpuBundle.GetDevice().GetQueue();

    // [1] BRICK GRID CELL
    wgpu::BufferDescriptor desc{};
    desc.size = numBricks * sizeof(BrickGridCell); // a single uint32_t pointer per cell
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    desc.label = "Brick Grid Buffer";
    desc.mappedAtCreation = false;
    wgpuBundle.SafeCreateBuffer(&desc, this->brickGridBuffer);
    queue.WriteBuffer(brickGridBuffer, 0, brickGrid.data(), desc.size);

    // [2] BRICK POOL BUFFER
    // The max number of bricks, is given by the Maximum Voxel Resolution divided by 8 (brick size)
    desc.size = numVisibleBricks * 16 * sizeof(uint32_t); // 8x8x8 occupancy per visible brick, times MAX_GPU_BRICKS
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    desc.label = "Brick Pool Buffer";
    desc.mappedAtCreation = false;
    wgpuBundle.SafeCreateBuffer(&desc, this->brickPoolBuffer);

    desc.size = numBricks * sizeof(uint32_t);
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    desc.label = "Brick Request Flags Buffer";
    desc.mappedAtCreation = false;
    wgpuBundle.SafeCreateBuffer(&desc, this->brickRequestFlagsBuffer);

    std::vector<uint32_t> zeroFlags(numBricks, 0);
    desc.size = numBricks * sizeof(uint32_t);
    desc.usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc;
    desc.label = "Brick Request Flags Reset Buffer";
    desc.mappedAtCreation = true;
    wgpuBundle.SafeCreateBuffer(&desc, this->brickRequestFlagsRESET);
    memcpy(brickRequestFlagsRESET.GetMappedRange(), zeroFlags.data(), numBricks * sizeof(uint32_t));
    brickRequestFlagsRESET.Unmap();

    // [3] COLOR POOL BUFFERS
    // 512 voxels * 3 bytes per voxel
    this->colorPoolBuffers.resize(MAX_COLOR_POOLS);
    uint64_t poolSize = this->maxColorBufferEntries * sizeof(uint32_t);
    uint64_t totalColorSizeNeeded = static_cast<uint64_t>(numVisibleBricks) * static_cast<uint64_t>(COLOR_BYTES_PER_BRICK);
    uint64_t remaining = totalColorSizeNeeded; // total size needed across all pools

    for (uint32_t i = 0; i < MAX_COLOR_POOLS; ++i)
    {
        uint64_t bufferSize = 0;

        if (i < numberOfColorPools && this->hasColor)
        {
            bufferSize = std::min(poolSize, remaining);

            if (bufferSize < sizeof(uint32_t))
                bufferSize = sizeof(uint32_t);

            remaining -= std::min(poolSize, remaining);

            desc.size  = bufferSize;
            std::cout << "[VoxelManager] Creating Color Pool Buffer " << i << " of size " << desc.size / 1024 << " KB\n";
            desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
            desc.label =
                ("Color Pool Buffer " + std::to_string(i)).c_str();
            desc.mappedAtCreation = false;

            wgpuBundle.SafeCreateBuffer(&desc, colorPoolBuffers[i]);
        }
        else
        {
            desc.size  = sizeof(uint32_t);
            desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
            desc.label =
                ("Dummy Color Pool Buffer " + std::to_string(i)).c_str();
            desc.mappedAtCreation = false;

            wgpuBundle.SafeCreateBuffer(&desc, colorPoolBuffers[i]);
        }
    }
}

//================================//
void VoxelManager::createUploadBindGroup(RenderPipelineWrapper& pipelineWrapper, WgpuBundle& wgpuBundle)
{
    // entries: 6, since max color pools is 3
    wgpu::BindGroupEntry entries[6]{};

    entries[0].binding = 0;
    entries[0].buffer = this->uploadBuffer;
    entries[0].offset = 0;
    entries[0].size = this->uploadBuffer.GetSize();

    entries[1].binding = 1;
    entries[1].buffer = this->uploadCountUniform;
    entries[1].offset = 0;
    entries[1].size = this->uploadCountUniform.GetSize();

    entries[2].binding = 2;
    entries[2].buffer = this->brickPoolBuffer;
    entries[2].offset = 0;
    entries[2].size = this->brickPoolBuffer.GetSize();

    for (int i = 0; i < MAX_COLOR_POOLS; ++i)
    {
        entries[3 + i].binding = 3 + i;
        entries[3 + i].buffer = this->colorPoolBuffers[i];
        entries[3 + i].offset = 0;
        entries[3 + i].size = this->colorPoolBuffers[i].GetSize();
    }

    wgpu::BindGroupDescriptor bindGroupDesc{};
    bindGroupDesc.layout = pipelineWrapper.bindGroupLayout;
    bindGroupDesc.entryCount = 6;
    bindGroupDesc.entries = entries;

    pipelineWrapper.bindGroup = wgpuBundle.GetDevice().CreateBindGroup(&bindGroupDesc);
}

//================================//
void VoxelManager::loadFile(const std::string& filename)
{
    // Does file exist?
    std::ifstream fileCheck(filename, std::ios::binary);
    if (!fileCheck)
    {
        std::cout << "[VoxelManager] Voxel file " << filename << " does not exist." << std::endl;
        return;
    }

    this->voxelFileReader = std::make_unique<VoxelFileReader>(filename);
    this->loadedMesh = true;

    std::cout << "[VoxelManager] Voxel file " << filename << " loaded. Resolution: " << this->voxelFileReader->getResolution() << std::endl;
}