#include "../includes/VoxelManager.hpp"
#include <iostream>

//================================//
// Bit layout (example):
// [31]     : resident flag
// [30]     : requested flag (GPU feedback)
// [29:24]  : unused
// [23:0]   : brick pool index OR packed LOD color
static uint32_t PackLOD(ColorRGB c)
{
    return (0u << 31) |
           (0u << 30) |
           (uint32_t(c.r) << 16) |
           (uint32_t(c.g) << 8)  |
           uint32_t(c.b);
}

//================================//
static uint32_t PackResident(uint32_t brickIndex)
{
    return (1u << 31) | (brickIndex & 0x00FFFFFFu);
}

//================================//
ColorRGB VoxelManager::computeBrickAverageColor(const BrickMapCPU& brick)
{
    uint64_t count = 0;
    uint64_t r = 0, g = 0, b = 0;

    for (int z = 0; z < 8; ++z)
    {
        uint64_t slice = brick.occupancy[z];
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

    if (filled)
    {
        brick.occupancy[lz] |= (1ull << bit);
        brick.colors[lz * 64 + bit] = color;
    }
    else
    {
        brick.occupancy[lz] &= ~(1ull << bit);
    }

    brick.lodColor = computeBrickAverageColor(brick); // Recompute LOD color average
    brick.dirty = true;
}

//================================//
void VoxelManager::update(wgpu::Queue& queue, wgpu::CommandEncoder& encoder)
{
    pendingUploadCount = 0;

    // Map the CPU upload buffer (MapWrite | CopySrc)
    UploadEntry* uploads = static_cast<UploadEntry*>(CPUuploadBuffer.GetMappedRange());

    for (uint32_t brickGridIndex = 0; brickGridIndex < brickMaps.size(); ++brickGridIndex)
    {
        BrickMapCPU& brick = brickMaps[brickGridIndex];

        if (!brick.dirty)
            continue;

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

        if (pendingUploadCount + 1 >= MAX_FEEDBACK) break;

        UploadEntry& entry = uploads[pendingUploadCount++];
        entry.brickGridIndex = brickGridIndex;
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
void VoxelManager::readFeedback(WgpuBundle& wgpuBundle)
{
    // TODO read the feedback buffer from GPU and process requested bricks
}

//================================//
void VoxelManager::initBuffers(WgpuBundle& wgpuBundle)
{
    std::cout << "[VoxelManager] Initializing buffers on CPU side...\n";
    // CPU storage initialization
    this->brickMaps.resize(this->BrickResolution * this->BrickResolution * this->BrickResolution);
    for (BrickMapCPU& brick : this->brickMaps)
    {
        std::memset(brick.occupancy, 0, sizeof(brick.occupancy));
        std::memset(brick.colors, 0, sizeof(brick.colors));
        brick.lodColor = {0, 0, 0};
        brick.dirty = false;
        brick.onGPU = false;
        brick.gpuBrickIndex = UINT32_MAX;
    }

    this->brickGrid.resize(this->BrickResolution * this->BrickResolution * this->BrickResolution);
    for(BrickGridCell& cell : this->brickGrid)
    {
        cell.pointer = PackLOD({0,0,0}); // Initialize to empty
    }

    this->freeBrickSlots.reserve(MAX_GPU_BRICKS);
    for (uint32_t i = 0; i < MAX_GPU_BRICKS; ++i)
        freeBrickSlots.push_back(i);


    std::cout << "[VoxelManager] Initializing buffers on GPU side...\n";
    // GPU storage initialization
    wgpu::Queue queue = wgpuBundle.GetDevice().GetQueue();

    // [1] BRICK GRID CELL
    wgpu::BufferDescriptor desc{};
    desc.size = brickGrid.size() * sizeof(BrickGridCell);
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    desc.label = "Brick Grid Buffer";
    desc.mappedAtCreation = false;
    this->brickGridBuffer = wgpuBundle.GetDevice().CreateBuffer(&desc);
    queue.WriteBuffer(brickGridBuffer, 0, brickGrid.data(), desc.size);

    // [2] BRICK POOL BUFFER
    // The max number of bricks, is given by the Maximum Voxel Resolution divided by 8 (brick size)
    std::vector<uint64_t> empty(MAX_GPU_BRICKS * 8, 0);
    desc.size = MAX_GPU_BRICKS * 8 * sizeof(uint64_t);
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    desc.label = "Brick Pool Buffer";
    desc.mappedAtCreation = false;
    this->brickPoolBuffer = wgpuBundle.GetDevice().CreateBuffer(&desc);
    queue.WriteBuffer(brickPoolBuffer, 0, empty.data(), desc.size);

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
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
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

    // CPU feedback count buffer
    desc.size = sizeof(uint32_t);
    desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    desc.label = "Feedback Count Buffer (CPU)";
    desc.mappedAtCreation = false;
    this->CPUfeedbackCountBuffer = wgpuBundle.GetDevice().CreateBuffer(&desc);

    // now the feedback indices buffer
    desc.size = MAX_FEEDBACK * sizeof(uint32_t);
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    desc.label = "Feedback Indices Buffer (GPU)";
    desc.mappedAtCreation = false;
    this->feedbackIndicesBuffer = wgpuBundle.GetDevice().CreateBuffer(&desc);

    // CPU feedback indices buffer
    desc.size = MAX_FEEDBACK * sizeof(uint32_t);
    desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    desc.label = "Feedback Indices Buffer (CPU)";
    desc.mappedAtCreation = false;
    this->CPUfeedbackIndicesBuffer = wgpuBundle.GetDevice().CreateBuffer(&desc);

    // Upload buffer initialization
    desc.size = MAX_FEEDBACK * sizeof(UploadEntry);
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    desc.label = "Upload Buffer (GPU)";
    desc.mappedAtCreation = false;
    this->uploadBuffer = wgpuBundle.GetDevice().CreateBuffer(&desc);

    // CPU upload buffer
    desc.size = MAX_FEEDBACK * sizeof(UploadEntry);
    desc.usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc;
    desc.label = "Upload Buffer (CPU)";
    desc.mappedAtCreation = true;
    this->CPUuploadBuffer = wgpuBundle.GetDevice().CreateBuffer(&desc);
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
    entries[3].buffer = this->brickGridBuffer;
    entries[3].offset = 0;
    entries[3].size = this->brickGridBuffer.GetSize();

    wgpu::BindGroupDescriptor bindGroupDesc{};
    bindGroupDesc.layout = pipelineWrapper.bindGroupLayout;
    bindGroupDesc.entryCount = 4;
    bindGroupDesc.entries = entries;

    pipelineWrapper.bindGroup = wgpuBundle.GetDevice().CreateBindGroup(&bindGroupDesc);
}