#ifndef VOXELIZER_HPP
#define VOXELIZER_HPP

#include "Rendering/wgpuBundle.hpp"
#include "Rendering/Pipelines/pipelines.hpp"
#include <algorithm>
#include <vector>
#include <array>
#include <fstream>
#include <string>
#include <bit>

// Forward declarations
class VoxelFileWriter;
class VoxelFileReader;

//================================//
struct VoxelizerUniforms
{
    uint32_t voxelResolution;
    uint32_t brickResolution;
    float    voxelSize;
    uint32_t numTriangles;
    float    meshMinBounds[3];
    uint32_t _pad1;
    uint32_t brickStart;
    uint32_t brickEnd;
    uint32_t _pad2[2];
};

struct Vertex
{
    float position[3];
    float _pad;
    float uv[2];
    float _pad2[2];
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
    void checkLimits(uint32_t& voxelResolution, uint32_t& maxBricksPerPass, uint8_t& numPasses);
    bool voxelizeMesh(const std::string& outputVoxelFile, uint32_t voxelResolution, uint32_t maxBricksPerPass, uint8_t numPasses);

private:

    void initializeGpuResources(uint32_t maxBricksPerPass);

    std::unique_ptr<WgpuBundle> gpuBundle;

    std::vector<std::array<double, 3>> verticesVec;
    std::vector<std::array<int, 3>> facesVec;
    std::vector<std::array<double, 3>> normalsVec;
    std::vector<std::array<double, 2>> uvsVec;


    bool hasTexture = false;
    unsigned char* textureData = nullptr;
    int texWidth, texHeight, texChannels;

    double meshWidth; // x axis extent
    double meshHeight; // y axis extent
    double meshDepth; // z axis extent
    std::array<double, 3> meshMinBounds;
    std::array<double, 3> meshMaxBounds;

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

    wgpu::Buffer counterReadbackBuffer;
    wgpu::Buffer occupancyReadbackBuffer;
    wgpu::Buffer brickOutputReadbackBuffer;
    wgpu::Buffer packedColorReadbackBuffer;

    RenderPipelineWrapper voxelizationPipeline;
    RenderPipelineWrapper compactVoxelPipeline;
};

#endif