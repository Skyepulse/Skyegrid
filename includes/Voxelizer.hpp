#ifndef VOXELIZER_HPP
#define VOXELIZER_HPP

#include "Rendering/wgpuBundle.hpp"
#include "Rendering/Pipelines/pipelines.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <vector>
#include <fstream>
#include <string>
#include <bit>

// Forward declarations
class VoxelFileWriter;
class VoxelFileReader;

//================================//
struct VoxelFileHeader
{
    uint32_t magic;
    uint32_t version;
    uint32_t resolution;
    uint32_t brickResolution;
    uint32_t numBricks;
    uint32_t occupiedBricks;
    uint64_t brickIndexOffset;
    uint64_t brickDataOffset;
    uint8_t  reserved[32];
};

struct brickIndexEntry
{
    uint32_t brickGridIndex; //(x + y*res + z*res^2)
    uint8_t LOD_R;
    uint8_t LOD_G;
    uint8_t LOD_B;
    uint8_t FLAGS;
    uint64_t dataOffset; // Offset to detailed data
    uint32_t dataSize;  // Size of detailed data
    uint32_t reserved; // Padding
};

struct VoxelColorRGB
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct brickDataEntry
{
    uint32_t occupancy[16]; // 8 slices of 8x8 occupancy
    std::vector<VoxelColorRGB> colors;
};

//================================//
struct VoxelizerUniforms
{
    uint32_t voxelResolution;
    uint32_t brickResolution;
    float    voxelSize;
    uint32_t numTriangles;
    float    meshMinBounds[3];
    uint32_t _pad;
};

struct Vertex
{
    float position[3];
    float uv[2];
    float normal[3];
    float padding;
};

struct Triangle
{
    uint32_t indices[3];
    uint32_t _pad;
};

//================================//
class Voxelizer
{
public:
    Voxelizer();
    ~Voxelizer();

    bool loadMesh(const std::string& filename, const std::string& texturePath = "");
    bool voxelizeMesh(const std::string& outputVoxelFile, uint32_t voxelResolution);

private:

    void initializeGpuResources(uint32_t voxelResolution);

    std::unique_ptr<WgpuBundle> gpuBundle;

    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
    Eigen::MatrixXi edges;
    Eigen::MatrixXd Normals;
    Eigen::MatrixXd UV;

    bool hasTexture = false;
    unsigned char* textureData = nullptr;
    int texWidth, texHeight, texChannels;

    double meshWidth; // x axis extent
    double meshHeight; // y axis extent
    double meshDepth; // z axis extent
    Eigen::Vector3d meshMinBounds;
    Eigen::Vector3d meshMaxBounds;

    // GPU resources
    wgpu::Buffer vertexBuffer;
    wgpu::Buffer triangleBuffer;
    wgpu::Buffer occupancyBuffer;
    wgpu::Buffer denseColorsBuffer;
    wgpu::Texture texture;
    wgpu::TextureView textureView;
    wgpu::Sampler textureSampler;

    wgpu::Buffer brickOutputBuffer;
    wgpu::Buffer packedColorBuffer;
    wgpu::Buffer countersBuffer;

    RenderPipelineWrapper voxelizationPipeline;
    RenderPipelineWrapper compactVoxelPipeline;
};

//================================//
class VoxelFileWriter
{
public:
    VoxelFileWriter(const std::string& filename, uint32_t resolution)
    {
        file.open(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Failed to create voxel file");

        // Write the hader first
        header.magic = 0x4C584F56; // 'VOXL' in little-endian
        header.version = 1;
        header.resolution = resolution;
        header.brickResolution = resolution / 8;
        header.numBricks = header.brickResolution * header.brickResolution * header.brickResolution;
        header.occupiedBricks = 0;
        header.brickIndexOffset = sizeof(VoxelFileHeader);
        header.brickDataOffset = 0;
        std::memset(header.reserved, 0, sizeof(header.reserved));

        file.write(reinterpret_cast<const char*>(&header), sizeof(VoxelFileHeader));
    }

    void AddBrick(uint32_t brickGridIndex, const uint32_t occupancy[16], const std::vector<VoxelColorRGB>& colors, VoxelColorRGB lodColor, uint8_t FLAGS=0)
    {
        brickIndexEntry indexEntry;
        indexEntry.brickGridIndex = brickGridIndex;
        indexEntry.LOD_R = lodColor.r;
        indexEntry.LOD_G = lodColor.g;
        indexEntry.LOD_B = lodColor.b;
        indexEntry.FLAGS = FLAGS;
        indexEntry.dataOffset = currentDataOffset;
        indexEntry.dataSize = 64 + colors.size() * 3;
        indexEntry.reserved = 0;
        brickIndex.push_back(indexEntry);

        brickDataEntry dataEntry;
        std::memcpy(dataEntry.occupancy, occupancy, 64);
        dataEntry.colors = colors;
        brickDataEntries.push_back(dataEntry);

        currentDataOffset += indexEntry.dataSize;
        align();
    }

    void EndFile()
    {
        // We want to write sorted, even if they were added out of order
        std::sort(brickIndex.begin(), brickIndex.end(), [](const brickIndexEntry& a, const brickIndexEntry& b) {
            return a.brickGridIndex < b.brickGridIndex;
        });

        file.seekp(sizeof(VoxelFileHeader), std::ios::beg); // We know we start just after header
        file.write(reinterpret_cast<const char*>(brickIndex.data()), brickIndex.size() * sizeof(brickIndexEntry));

        header.brickDataOffset = header.brickIndexOffset + brickIndex.size() * sizeof(brickIndexEntry);

        file.seekp(header.brickDataOffset,  std::ios::beg);
        for (size_t i = 0; i < brickDataEntries.size(); ++i)
        {
            const brickDataEntry& dataEntry = brickDataEntries[i];
            file.write(reinterpret_cast<const char*>(dataEntry.occupancy), 64);
            file.write(reinterpret_cast<const char*>(dataEntry.colors.data()), dataEntry.colors.size() * 3);

            // what position are we at?
            uint64_t pos = file.tellp();
            if (pos % 4 != 0)
            {
                uint32_t pad = 0;
                file.write(reinterpret_cast<const char*>(&pad), 4 - (pos % 4));
            }
        }

        header.occupiedBricks = static_cast<uint32_t>(brickIndex.size());
        // Rewrite header with updated info
        file.seekp(0, std::ios::beg);
        file.write(reinterpret_cast<const char*>(&header), sizeof(VoxelFileHeader));
    }

private:

    inline void align()
    {
        if (currentDataOffset % 4 != 0)
            currentDataOffset += (4 - currentDataOffset % 4);
    }

    std::ofstream file;
    VoxelFileHeader header;
    std::vector<brickIndexEntry> brickIndex;
    std::vector<brickDataEntry> brickDataEntries;
    uint64_t currentDataOffset = 0;
};

//================================//
class VoxelFileReader
{
public:
    VoxelFileReader(const std::string& filename)
    {
        file.open(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Failed to open voxel file");

        // Read header
        file.read(reinterpret_cast<char*>(&header), sizeof(VoxelFileHeader));
        if (header.magic != 0x4C584F56) // 'VOXL' in little-endian
            throw std::runtime_error("Invalid voxel file format");

        // Read brick index
        brickIndex.resize(header.occupiedBricks);
        file.seekg(header.brickIndexOffset, std::ios::beg);
        
        // This way we read all the indices of the occupied bricks on the fly
        file.read(reinterpret_cast<char*>(brickIndex.data()), header.occupiedBricks * sizeof(brickIndexEntry));
    }

    bool IsBrickOccupied(uint32_t brickGridIndex) const
    {
        auto it = std::lower_bound(brickIndex.begin(), brickIndex.end(), brickGridIndex,
            [](const brickIndexEntry& entry, uint32_t idx) {
                return entry.brickGridIndex < idx;
            });
        
        return (it != brickIndex.end() && it->brickGridIndex == brickGridIndex);
    }

    bool getBrickData(uint32_t brickGridIndex, brickDataEntry& outData) const
    {
        // BINARY SEARCH
        auto it = std::lower_bound(brickIndex.begin(), brickIndex.end(), brickGridIndex,
            [](const brickIndexEntry& entry, uint32_t idx) 
            {
                return entry.brickGridIndex < idx;
            });
        
        if (it == brickIndex.end() || it->brickGridIndex != brickGridIndex)
            return false;

        // Read brick data
        file.seekg(header.brickDataOffset + it->dataOffset, std::ios::beg);
        file.read(reinterpret_cast<char*>(outData.occupancy), 64);

        uint32_t occupiedVoxels = 0;
        for (int i = 0; i < 16; ++i)
            occupiedVoxels += std::popcount(outData.occupancy[i]); // per documentation in C++20, counts the number of set bits in an integer

        outData.colors.resize(occupiedVoxels);
        file.read(reinterpret_cast<char*>(outData.colors.data()), occupiedVoxels * 3);

        return true;
    }

    void getInitialOccupiedBricks(std::vector<brickIndexEntry>& outBricks) const
    {
        outBricks = brickIndex;
    }

    uint32_t getResolution() const { return header.resolution; }
    
private:
    // we use mutable because seekg changes internal state of file stream
    mutable std::ifstream file;
    VoxelFileHeader header;
    std::vector<brickIndexEntry> brickIndex;
};

#endif