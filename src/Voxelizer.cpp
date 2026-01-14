#include "../includes/Voxelizer.hpp"

#include <igl/readPLY.h>
#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <iostream>
#include <filesystem>
#include <Eigen/Geometry>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" 

const double EPSILON = 1e-10;

//================================//
struct BrickOutput {
    uint32_t brickGridIndex;
    uint32_t lodColor;
    uint32_t dataOffset;
    uint32_t numOccupied;
};

//================================//
Voxelizer::Voxelizer()
{
    this->gpuBundle = std::make_unique<WgpuBundle>(WindowFormat{nullptr, 1, 1, false});
    this->textureData = nullptr;

    CreateVoxelizationPipeline(*this->gpuBundle, this->voxelizationPipeline);
    CreateCompactVoxelPipeline(*this->gpuBundle, this->compactVoxelPipeline);
}

//================================//
Voxelizer::~Voxelizer()
{
    if (textureData) 
    {
        stbi_image_free(textureData);
    }
}

//================================//
static bool safeTextureLoad(const std::string& texturePath, unsigned char** textureData, int* width, int* height, int* channels)
{
    try 
    {
        if (!std::filesystem::exists(texturePath))
            return false;
    }
    catch (const std::filesystem::filesystem_error&) 
    {
        std::cout << "[Voxelizer] Filesystem error when checking texture path: " << texturePath << std::endl;
        return false;
    }

    *textureData = stbi_load(texturePath.c_str(), width, height, channels, 0);
    return (*textureData != nullptr);
}

//================================//
bool Voxelizer::loadMesh(const std::string& filename, const std::string& texturePath)
{
    this->vertices.resize(0, 0);
    this->faces.resize(0, 0);
    this->edges.resize(0, 0);
    this->Normals.resize(0, 0);
    this->UV.resize(0, 0);

    // read triangle mesh and colors if available
    std::string term = filename.substr(filename.find_last_of('.') + 1);

    bool success = false;
    if (term == "ply")
        success = igl::readPLY(filename, this->vertices, this->faces, this->edges, this->Normals, this->UV);
    else if (term == "obj")
    {
        Eigen::MatrixXi FTC;
        Eigen::MatrixXi FN;
        success = igl::readOBJ(filename, this->vertices, this->UV, this->Normals, this->faces, FTC, FN);
    }
    else if (term == "off")
    {
        std::vector<std::vector<double>> vV;
        std::vector<std::vector<int>> vF;
        std::vector<std::vector<double>> vN;
        std::vector<std::vector<double>> vUV;
        success = igl::readOFF(filename, vV, vF, vN, vUV);

        if (success)
        {
            // Convert to Eigen matrices
            this->vertices.resize(vV.size(), 3);
            for (size_t i = 0; i < vV.size(); ++i)
            {
                this->vertices(i, 0) = vV[i][0];
                this->vertices(i, 1) = vV[i][1];
                this->vertices(i, 2) = vV[i][2];
            }

            this->faces.resize(vF.size(), 3);
            for (size_t i = 0; i < vF.size(); ++i)
            {
                this->faces(i, 0) = vF[i][0];
                this->faces(i, 1) = vF[i][1];
                this->faces(i, 2) = vF[i][2];
            }

            if (!vN.empty())
            {
                this->Normals.resize(vN.size(), 3);
                for (size_t i = 0; i < vN.size(); ++i)
                {
                    this->Normals(i, 0) = vN[i][0];
                    this->Normals(i, 1) = vN[i][1];
                    this->Normals(i, 2) = vN[i][2];
                }
            }

            if (!vUV.empty())
            {
                this->UV.resize(vUV.size(), 2);
                for (size_t i = 0; i < vUV.size(); ++i)
                {
                    this->UV(i, 0) = vUV[i][0];
                    this->UV(i, 1) = vUV[i][1];
                }
            }
        }
    }

    if (success)
    {
        std::cout << "[Voxelizer] Successfully loaded mesh from " << filename << " with "
                  << this->vertices.rows() << " vertices and "
                  << this->faces.rows() << " faces" << std::endl;
    }
    else
    {
        std::cout << "[Voxelizer] Failed to load mesh from " << filename << std::endl;
        return false;
    }

    // Get extents and min bounds
    this->meshMinBounds = this->vertices.colwise().minCoeff();
    this->meshMaxBounds = this->vertices.colwise().maxCoeff();
    this->meshWidth = this->meshMaxBounds.x() - this->meshMinBounds.x();
    this->meshHeight = this->meshMaxBounds.y() - this->meshMinBounds.y();
    this->meshDepth = this->meshMaxBounds.z() - this->meshMinBounds.z();

    // READ TEXTURE
    if(this->textureData != nullptr)
    {
        stbi_image_free(this->textureData);
        this->textureData = nullptr;
    }

    if (!texturePath.empty())
    {
        this->hasTexture = safeTextureLoad(texturePath, &this->textureData, &this->texWidth, &this->texHeight, &this->texChannels);
    }
    else
    {
        std::string defaultTexturePath = filename.substr(0, filename.find_last_of('.')) + ".png";
        bool success = this->hasTexture = safeTextureLoad(defaultTexturePath, &this->textureData, &this->texWidth, &this->texHeight, &this->texChannels);

        if (!success)
        {
            defaultTexturePath = filename.substr(0, filename.find_last_of('.')) + ".jpg";
            this->hasTexture = safeTextureLoad(defaultTexturePath, &this->textureData, &this->texWidth, &this->texHeight, &this->texChannels);
        }
    }

    if (this->hasTexture)
    {
        std::cout << "[Voxelizer] Successfully loaded texture with size "
                    << this->texWidth << "x" << this->texHeight << " and "
                    << this->texChannels << " channels." << std::endl;
    }
    else
    {
        std::cout << "[Voxelizer] Failed to load texture for the mesh." << std::endl;
    }

    return success;
}

//================================//
void Voxelizer::initializeGpuResources(uint32_t maxBricksPerPass)
{
    std::vector<Vertex> vertexData;
    std::vector<Triangle> triangleData;

    wgpu::BufferDescriptor  bufferDesc{};
    wgpu::Queue queue = this->gpuBundle->GetDevice().GetQueue();

    // [1] vertex data
    vertexData.resize(this->vertices.rows());
    for(int i = 0; i < this->vertices.rows(); i++)
    {
        Vertex v;

        v.position[0] = static_cast<float>(this->vertices(i, 0));
        v.position[1] = static_cast<float>(this->vertices(i, 1));
        v.position[2] = static_cast<float>(this->vertices(i, 2));

        if(this->UV.rows() > 0 && i < this->UV.rows())
        {
            v.uv[0] = static_cast<float>(this->UV(i, 0));
            v.uv[1] = static_cast<float>(this->UV(i, 1));
        }
        else
        {
            v.uv[0] = 0.0f;
            v.uv[1] = 0.0f;
        }

        if(this->Normals.rows() > 0 && i < this->Normals.rows())
        {
            v.normal[0] = static_cast<float>(this->Normals(i, 0));
            v.normal[1] = static_cast<float>(this->Normals(i, 1));
            v.normal[2] = static_cast<float>(this->Normals(i, 2));
        }
        else
        {
            v.normal[0] = 0.0f;
            v.normal[1] = 1.0f;
            v.normal[2] = 0.0f;
        }
        v.padding = 0.0f;
        vertexData[i] = v;
    }

    bufferDesc.size = sizeof(Vertex) * vertexData.size();
    bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    bufferDesc.mappedAtCreation = false;
    bufferDesc.label = "Vertex Buffer";
    this->gpuBundle->SafeCreateBuffer(&bufferDesc, this->vertexBuffer);
    queue.WriteBuffer(this->vertexBuffer, 0, vertexData.data(), bufferDesc.size);

    // [2] triangle data
    triangleData.resize(this->faces.rows());
    for(int i = 0; i < this->faces.rows(); i++)
    {
        Triangle t;
        t.indices[0] = static_cast<uint32_t>(this->faces(i, 0));
        t.indices[1] = static_cast<uint32_t>(this->faces(i, 1));
        t.indices[2] = static_cast<uint32_t>(this->faces(i, 2));
        t._pad = 0;
        triangleData[i] = t;
    }

    bufferDesc.size = sizeof(Triangle) * triangleData.size();
    bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    bufferDesc.mappedAtCreation = false;
    bufferDesc.label = "Triangle Buffer";
    this->gpuBundle->SafeCreateBuffer(&bufferDesc, this->triangleBuffer);
    queue.WriteBuffer(this->triangleBuffer, 0, triangleData.data(), bufferDesc.size);

    // [3] Occupancy
    bufferDesc.size = sizeof(uint32_t) * 16 * maxBricksPerPass;
    bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
    bufferDesc.mappedAtCreation = false;
    bufferDesc.label = "Occupancy Buffer";
    this->gpuBundle->SafeCreateBuffer(&bufferDesc, this->occupancyBuffer);
    std::vector<uint32_t> zeros(16 * maxBricksPerPass, 0);
    queue.WriteBuffer(this->occupancyBuffer, 0, zeros.data(), bufferDesc.size);

    // [4] Dense Colors
    bufferDesc.size = sizeof(uint32_t) * maxBricksPerPass * 512;
    bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
    bufferDesc.mappedAtCreation = false;
    bufferDesc.label = "Dense Colors Buffer";
    this->gpuBundle->SafeCreateBuffer(&bufferDesc, this->denseColorsBuffer);

    // [5] brick output buffer
    bufferDesc.size = sizeof(BrickOutput) * maxBricksPerPass;
    bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    bufferDesc.mappedAtCreation = false;
    bufferDesc.label = "Brick Output Buffer";
    this->gpuBundle->SafeCreateBuffer(&bufferDesc, this->brickOutputBuffer);

    // [6] packed color buffer
    bufferDesc.size = sizeof(uint32_t) * maxBricksPerPass * 512;
    bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    bufferDesc.mappedAtCreation = false;
    bufferDesc.label = "Packed Color Buffer";
    this->gpuBundle->SafeCreateBuffer(&bufferDesc, this->packedColorBuffer);

    // [7] atomic counters
    bufferDesc.size = sizeof(uint32_t) * 2;
    bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
    bufferDesc.mappedAtCreation = false;
    bufferDesc.label = "Counters Buffer";
    this->gpuBundle->SafeCreateBuffer(&bufferDesc, this->countersBuffer);
    uint32_t counterZeros[2] = {0, 0};
    queue.WriteBuffer(this->countersBuffer, 0, counterZeros, sizeof(counterZeros));

    //[8] readback buffers, at worst case scenario size initialization with maxBricksPerPass
    bufferDesc.size = sizeof(uint32_t) * 2;
    bufferDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    bufferDesc.mappedAtCreation = false;
    bufferDesc.label = "Counter Readback Buffer";
    this->gpuBundle->SafeCreateBuffer(&bufferDesc, this->counterReadbackBuffer);

    bufferDesc.size = sizeof(uint32_t) * 16 * maxBricksPerPass;
    bufferDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    bufferDesc.mappedAtCreation = false;
    bufferDesc.label = "Occupancy Readback Buffer";
    this->gpuBundle->SafeCreateBuffer(&bufferDesc, this->occupancyReadbackBuffer);

    uint32_t occupiedBrickCount = maxBricksPerPass;
    bufferDesc.size = sizeof(BrickOutput) * occupiedBrickCount;
    bufferDesc.label = "Brick Output Readback Buffer";
    this->gpuBundle->SafeCreateBuffer(&bufferDesc, this->brickOutputReadbackBuffer);

    uint32_t totalColorCount = maxBricksPerPass * 512;
    bufferDesc.size = sizeof(uint32_t) * totalColorCount;;
    bufferDesc.label = "Packed Color Readback Buffer";
    this->gpuBundle->SafeCreateBuffer(&bufferDesc, this->packedColorReadbackBuffer);

    // [8] texture, texture view, sampler
    if (this->hasTexture && this->textureData)
    {
        wgpu::TextureDescriptor textureDesc{};
        textureDesc.size = {static_cast<uint32_t>(this->texWidth), static_cast<uint32_t>(this->texHeight), 1};
        textureDesc.mipLevelCount = 1;
        textureDesc.sampleCount = 1;
        textureDesc.dimension = wgpu::TextureDimension::e2D;
        textureDesc.format = wgpu::TextureFormat::RGBA8Unorm;
        textureDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
        textureDesc.label = "Mesh Texture";

        this->texture = this->gpuBundle->GetDevice().CreateTexture(&textureDesc);
        std::vector<uint8_t> rgbaData;
        if (texChannels == 3) // Convert to RGBA, NEEDED IN COMPUTE SHADER
        {
            rgbaData.resize(this->texWidth * this->texHeight * 4);
            for(int i = 0; i < this->texWidth * this->texHeight; i++)
            {
                rgbaData[i * 4 + 0] = this->textureData[i * 3 + 0];
                rgbaData[i * 4 + 1] = this->textureData[i * 3 + 1];
                rgbaData[i * 4 + 2] = this->textureData[i * 3 + 2];
                rgbaData[i * 4 + 3] = 255;
            }
        }
        else if (texChannels == 4)
        {
            rgbaData.assign(this->textureData, this->textureData + (this->texWidth * this->texHeight * this->texChannels));
        }
        else if (texChannels == 1)
        {
            rgbaData.resize(this->texWidth * this->texHeight * 4);
            for(int i = 0; i < this->texWidth * this->texHeight; i++)
            {
                uint8_t value = this->textureData[i];
                rgbaData[i * 4 + 0] = value;
                rgbaData[i * 4 + 1] = value;
                rgbaData[i * 4 + 2] = value;
                rgbaData[i * 4 + 3] = 255;
            }
        }
        else
        {
            rgbaData.resize(texWidth * texHeight * 4, 255); // default white
        }

        wgpu::TexelCopyTextureInfo dstTexture{};
        dstTexture.texture = this->texture;
        dstTexture.mipLevel = 0;
        dstTexture.origin = {0, 0, 0};
        dstTexture.aspect = wgpu::TextureAspect::All;

        wgpu::TexelCopyBufferLayout srcBufferLayout{};
        srcBufferLayout.offset = 0;
        srcBufferLayout.bytesPerRow = this->texWidth * 4;
        srcBufferLayout.rowsPerImage = this->texHeight;

        wgpu::Extent3D writeSize{
            static_cast<uint32_t>(texWidth),
            static_cast<uint32_t>(texHeight),
            1
        };

        queue.WriteTexture(&dstTexture, rgbaData.data(), rgbaData.size(), &srcBufferLayout, &writeSize);

        wgpu::TextureViewDescriptor viewDesc{};
        viewDesc.format = wgpu::TextureFormat::RGBA8Unorm;
        viewDesc.dimension = wgpu::TextureViewDimension::e2D;
        viewDesc.baseMipLevel = 0;
        viewDesc.mipLevelCount = 1;
        viewDesc.baseArrayLayer = 0;
        viewDesc.arrayLayerCount = 1;
        viewDesc.aspect = wgpu::TextureAspect::All;
        viewDesc.label = "Mesh Texture View";
        this->textureView = this->texture.CreateView(&viewDesc);

        wgpu::SamplerDescriptor samplerDesc{};
        samplerDesc.addressModeU = wgpu::AddressMode::Repeat;
        samplerDesc.addressModeV = wgpu::AddressMode::Repeat;
        samplerDesc.addressModeW = wgpu::AddressMode::Repeat;
        samplerDesc.magFilter = wgpu::FilterMode::Linear;
        samplerDesc.minFilter = wgpu::FilterMode::Linear;
        samplerDesc.mipmapFilter = wgpu::MipmapFilterMode::Nearest;
        samplerDesc.lodMaxClamp = 1.0f;
        samplerDesc.lodMinClamp = 0.0f;
        samplerDesc.compare = wgpu::CompareFunction::Undefined;
        samplerDesc.maxAnisotropy = 1;
        samplerDesc.label = "Mesh Texture Sampler";

        this->textureSampler = this->gpuBundle->GetDevice().CreateSampler(&samplerDesc);
    }
    else // ALL WHITE TEXTURE BY DEFAULT
    {
        wgpu::TextureDescriptor textureDesc{};
        textureDesc.size = {1, 1, 1};
        textureDesc.mipLevelCount = 1;
        textureDesc.sampleCount = 1;
        textureDesc.dimension = wgpu::TextureDimension::e2D;
        textureDesc.format = wgpu::TextureFormat::RGBA8Unorm;
        textureDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
        this->texture = this->gpuBundle->GetDevice().CreateTexture(&textureDesc);

        uint8_t whitePixel[4] = {255, 255, 255, 255};

        wgpu::TexelCopyTextureInfo dstTexture{};
        dstTexture.texture = this->texture;
        dstTexture.mipLevel = 0;
        dstTexture.origin = {0, 0, 0};

        wgpu::TexelCopyBufferLayout srcBufferLayout{};
        srcBufferLayout.offset = 0;
        srcBufferLayout.bytesPerRow = 4;
        srcBufferLayout.rowsPerImage = 1;

        wgpu::Extent3D writeSize{1,1,1};

        queue.WriteTexture(&dstTexture, whitePixel, sizeof(whitePixel), &srcBufferLayout, &writeSize);
        this->textureView = this->texture.CreateView();

        wgpu::SamplerDescriptor samplerDesc{};
        samplerDesc.magFilter = wgpu::FilterMode::Nearest;
        samplerDesc.minFilter = wgpu::FilterMode::Nearest;
        this->textureSampler = this->gpuBundle->GetDevice().CreateSampler(&samplerDesc);
    }
}

//================================//
void Voxelizer::checkLimits(uint32_t& voxelResolution, uint32_t& maxBricksPerPass, uint8_t& numPasses)
{
    const uint64_t colorBytesPerBrick = sizeof(uint32_t) * 8 * 8 * 8;
    uint64_t maxBufferSize = this->gpuBundle->GetLimits().maxBufferSize * 0.6;
    uint64_t maxColorBufferSize = (maxBufferSize / static_cast<uint64_t>(colorBytesPerBrick)) * static_cast<uint64_t>(colorBytesPerBrick);

    if(voxelResolution <= 0) voxelResolution = 8;
    voxelResolution = (voxelResolution / 8) * 8; // ensure multiple of 8
    // Check if voxels are bigger than max num bricks encoding of 24 bits
    uint32_t maxBrickResolution = 1 << 8; // 256 bricks per axis max for 24 bit encoding
    uint32_t brickResolution = voxelResolution / 8;
    if(brickResolution > maxBrickResolution)
    {
        brickResolution = maxBrickResolution - 1;
        voxelResolution = brickResolution * 8;
        std::cout << "[Voxelizer] Warning: Voxel resolution too high, clamped to " << voxelResolution << std::endl;
    }

    uint64_t totalBricks = static_cast<uint64_t>(brickResolution) * brickResolution * brickResolution;
    uint64_t totalColorBufferSize = totalBricks * colorBytesPerBrick;
    if(totalColorBufferSize > maxColorBufferSize)
    {
        maxBricksPerPass = static_cast<uint32_t>(maxColorBufferSize / colorBytesPerBrick);
        std::cout << "[Voxelizer] Warning: Voxel resolution too high for available GPU memory, max bricks per pass set to " <<
            maxBricksPerPass << " (" << (maxBricksPerPass * 8) << "^3 voxels)" << std::endl;
        numPasses = static_cast<uint8_t>((totalBricks + maxBricksPerPass - 1) / maxBricksPerPass);
        std::cout << "[Voxelizer] Voxelization will be performed in " << static_cast<uint32_t>(numPasses) << " passes." << std::endl;
    }
    else
    {
        maxBricksPerPass = static_cast<uint32_t>(totalBricks);
        std::cout << "[Voxelizer] Voxelization can proceed with " << maxBricksPerPass << " bricks in only one pass." << std::endl;
        numPasses = 1;
    }
}

//================================//
bool Voxelizer::voxelizeMesh(const std::string& outputVoxelFile, uint32_t voxelResolution, uint32_t maxBricksPerPass, uint8_t numPasses)
{
    if (this->vertices.rows() == 0 || this->faces.rows() == 0) 
    {
        std::cerr << "[Voxelizer] No mesh loaded" << std::endl;
        return false;
    }

    std::cout << "[Voxelizer] Starting voxelization with resolution " << voxelResolution 
              << " (" << (voxelResolution * voxelResolution * voxelResolution) << " voxels)" << std::endl;
    std::cout << "[Voxelizer] Max bricks per pass: " << maxBricksPerPass << std::endl;
    std::cout << "[Voxelizer] Number of passes: " << static_cast<uint32_t>(numPasses) << std::endl;

    this->initializeGpuResources(maxBricksPerPass);

    wgpu::Device& device = this->gpuBundle->GetDevice();
    wgpu::Queue queue = device.GetQueue();

    double maxExtent = std::max({meshWidth, meshHeight, meshDepth});
    float voxelSize = static_cast<float>(maxExtent / voxelResolution);

    VoxelFileWriter writer(outputVoxelFile, voxelResolution);

    // Uniform
    VoxelizerUniforms uniforms;
    uniforms.voxelResolution = voxelResolution;
    uniforms.brickResolution = voxelResolution / 8;
    uniforms.voxelSize = voxelSize;
    uniforms.numTriangles = static_cast<uint32_t>(this->faces.rows());
    uniforms.meshMinBounds[0] = static_cast<float>(this->meshMinBounds.x());
    uniforms.meshMinBounds[1] = static_cast<float>(this->meshMinBounds.y());
    uniforms.meshMinBounds[2] = static_cast<float>(this->meshMinBounds.z());   
    uniforms._pad1 = 0;     
    uniforms._pad2[0] = 0;
    uniforms._pad2[1] = 0;

    wgpu::BufferDescriptor uniformBufferDesc{};
    uniformBufferDesc.size = sizeof(VoxelizerUniforms);
    uniformBufferDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniformBufferDesc.mappedAtCreation = false;
    uniformBufferDesc.label = "Voxelizer Uniform Buffer";
    wgpu::Buffer uniformBuffer;;
    this->gpuBundle->SafeCreateBuffer(&uniformBufferDesc, uniformBuffer);

    uint32_t bricksProcessed = 0;
    for (uint8_t pass = 0; pass < numPasses; pass++)
    {
        std::clock_t startPassTime = std::clock();

        uint32_t brickStart = bricksProcessed;
        uint32_t bricksThisPass = std::min(maxBricksPerPass, (voxelResolution / 8) * (voxelResolution / 8) * (voxelResolution / 8) - bricksProcessed);
        uint32_t brickEnd = brickStart + bricksThisPass;

        uniforms.brickStart = brickStart;
        uniforms.brickEnd = brickEnd;
        queue.WriteBuffer(uniformBuffer, 0, &uniforms, sizeof(VoxelizerUniforms));

        std::cout << "[Voxelizer] Pass " << (pass + 1) << "/" << (int)numPasses 
                  << ": Processing bricks " << brickStart << " to " << brickEnd - 1 << std::endl;

        // Reset counters
        uint32_t counterZeros[2] = {0, 0};
        queue.WriteBuffer(this->countersBuffer, 0, counterZeros, sizeof(counterZeros));

        // Clear occupancy buffer and color dense buffer for this pass
        std::vector<uint32_t> occZeros(16 * bricksThisPass, 0);
        queue.WriteBuffer(this->occupancyBuffer, 0, occZeros.data(), sizeof(uint32_t) * 16 * bricksThisPass);
        std::vector<uint32_t> colorZeros(512 * bricksThisPass, 0);
        queue.WriteBuffer(this->denseColorsBuffer, 0, colorZeros.data(), sizeof(uint32_t) * 512 * bricksThisPass);

        // [1] Voxelization pass
        {
            wgpu::BindGroupEntry entries[7]{};
            entries[0].binding = 0; entries[0].buffer = uniformBuffer; entries[0].size = sizeof(VoxelizerUniforms);
            entries[1].binding = 1; entries[1].buffer = this->vertexBuffer; entries[1].size = sizeof(Vertex) * vertices.rows();
            entries[2].binding = 2; entries[2].buffer = this->triangleBuffer; entries[2].size = sizeof(Triangle) * faces.rows();
            entries[3].binding = 3; entries[3].textureView = this->textureView;
            entries[4].binding = 4; entries[4].sampler = this->textureSampler;
            entries[5].binding = 5; entries[5].buffer = this->occupancyBuffer; entries[5].size = sizeof(uint32_t) * 16 * bricksThisPass;
            entries[6].binding = 6; entries[6].buffer = this->denseColorsBuffer; entries[6].size = sizeof(uint32_t) * bricksThisPass * 512;

            wgpu::BindGroupDescriptor bindGroupDesc{};
            bindGroupDesc.layout = this->voxelizationPipeline.bindGroupLayout;
            bindGroupDesc.entryCount = 7;
            bindGroupDesc.entries = entries;
            wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);

            wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
            wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
            pass.SetPipeline(this->voxelizationPipeline.computePipeline);
            pass.SetBindGroup(0, bindGroup);

            uint32_t numWorkGroups = (bricksThisPass + 63) / 64;
            pass.DispatchWorkgroups(numWorkGroups, 1, 1);
            pass.End();

            wgpu::CommandBuffer commandBuffer = encoder.Finish();
            queue.Submit(1, &commandBuffer);
        }

        // [2] Compact
        {
            wgpu::BindGroupEntry entries[6]{};
            entries[0].binding = 0; entries[0].buffer = uniformBuffer; entries[0].size = sizeof(VoxelizerUniforms);
            entries[1].binding = 1; entries[1].buffer = this->occupancyBuffer; entries[1].size = sizeof(uint32_t) * 16 * bricksThisPass;
            entries[2].binding = 2; entries[2].buffer = this->denseColorsBuffer; entries[2].size = sizeof(uint32_t) * bricksThisPass * 512;
            entries[3].binding = 3; entries[3].buffer = this->brickOutputBuffer; entries[3].size = sizeof(BrickOutput) * bricksThisPass;
            entries[4].binding = 4; entries[4].buffer = this->packedColorBuffer; entries[4].size = sizeof(uint32_t) * bricksThisPass * 512;
            entries[5].binding = 5; entries[5].buffer = this->countersBuffer; entries[5].size = sizeof(uint32_t) * 2;

            wgpu::BindGroupDescriptor bindGroupDesc{};
            bindGroupDesc.layout = this->compactVoxelPipeline.bindGroupLayout;
            bindGroupDesc.entryCount = 6;
            bindGroupDesc.entries = entries;
            wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);

            wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
            wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
            pass.SetPipeline(this->compactVoxelPipeline.computePipeline);
            pass.SetBindGroup(0, bindGroup);

            uint32_t numWorkGroups = (bricksThisPass + 63) / 64;
            pass.DispatchWorkgroups(numWorkGroups, 1, 1);
            pass.End();

            wgpu::CommandBuffer commandBuffer = encoder.Finish();
            queue.Submit(1, &commandBuffer);
        }

        // We can wait fer the GPU to finish here
        {
            wgpu::Future workDoneFuture = queue.OnSubmittedWorkDone(
                wgpu::CallbackMode::WaitAnyOnly,
                [](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
                    if (status != wgpu::QueueWorkDoneStatus::Success) {
                        std::cerr << "[Voxelizer] Queue work failed: " << std::string(message.data, message.length) << std::endl;
                    }
                }
            );
            this->gpuBundle->GetInstance().WaitAny(workDoneFuture, UINT64_MAX);
            this->gpuBundle->GetInstance().WaitAny(workDoneFuture, UINT64_MAX);
        }

        // Reset readback buffer descriptors
        {
            wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
            encoder.CopyBufferToBuffer(this->countersBuffer, 0, this->counterReadbackBuffer, 0, sizeof(uint32_t) * 2);
            wgpu::CommandBuffer commandBuffer = encoder.Finish();
            queue.Submit(1, &commandBuffer);
        }

        uint32_t occupiedBrickCount = 0;
        uint32_t totalColorCount = 0;
        {
            wgpu::Future mapFuture = this->counterReadbackBuffer.MapAsync(
                wgpu::MapMode::Read, 0, sizeof(uint32_t) * 2,
                wgpu::CallbackMode::WaitAnyOnly,
                [](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                    if (status != wgpu::MapAsyncStatus::Success) {
                        std::cerr << "[Voxelizer] Counter map failed: " << std::string(message.data, message.length) << std::endl;
                    }
                }
            );
            this->gpuBundle->GetInstance().WaitAny(mapFuture, UINT64_MAX);
            
            const uint32_t* data = static_cast<const uint32_t*>(this->counterReadbackBuffer.GetConstMappedRange(0, sizeof(uint32_t) * 2));
            if (!data) 
            {
                std::cerr << "[Voxelizer] Failed to get mapped range for counters" << std::endl;
                return false;
            }
            occupiedBrickCount = data[0];
            totalColorCount = data[1];
            this->counterReadbackBuffer.Unmap();
        }

        assert(occupiedBrickCount <= bricksThisPass);
        assert(totalColorCount <= bricksThisPass * 512);

        std::cout << "[Voxelizer] Pass " << (pass + 1) << ": " << occupiedBrickCount 
                  << " occupied bricks, " << totalColorCount << " colors" << std::endl;

                
        if (occupiedBrickCount == 0) 
        {
            bricksProcessed += bricksThisPass;
            continue;
        }

        // Prepare to read the data back, in this pass
        {
            wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
            encoder.CopyBufferToBuffer(this->occupancyBuffer, 0, this->occupancyReadbackBuffer, 0, sizeof(uint32_t) * 16 * bricksThisPass);
            encoder.CopyBufferToBuffer(this->brickOutputBuffer, 0, this->brickOutputReadbackBuffer, 0, sizeof(BrickOutput) * occupiedBrickCount);
            encoder.CopyBufferToBuffer(this->packedColorBuffer, 0, this->packedColorReadbackBuffer, 0, sizeof(uint32_t) * totalColorCount);
            wgpu::CommandBuffer commandBuffer = encoder.Finish();
            queue.Submit(1, &commandBuffer);
        }

        wgpu::Future occMapFuture = this->occupancyReadbackBuffer.MapAsync(
            wgpu::MapMode::Read, 0, sizeof(uint32_t) * 16 * bricksThisPass,
            wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                if (status != wgpu::MapAsyncStatus::Success) {
                    std::cerr << "[Voxelizer] Occupancy map failed: " << std::string(message.data, message.length) << std::endl;
                }
            }
        );

        wgpu::Future brickMapFuture = this->brickOutputReadbackBuffer.MapAsync(
            wgpu::MapMode::Read, 0, sizeof(BrickOutput) * occupiedBrickCount,
            wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                if (status != wgpu::MapAsyncStatus::Success) {
                    std::cerr << "[Voxelizer] Brick output map failed: " << std::string(message.data, message.length) << std::endl;
                }
            }
        );

        wgpu::Future colorMapFuture = this->packedColorReadbackBuffer.MapAsync(
            wgpu::MapMode::Read, 0, sizeof(uint32_t) * totalColorCount,
            wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                if (status != wgpu::MapAsyncStatus::Success) {
                    std::cerr << "[Voxelizer] Colors map failed: " << std::string(message.data, message.length) << std::endl;
                }
            }
        );

        // Wait for all three maps
        wgpu::FutureWaitInfo waitInfos[3] = {
            {occMapFuture, false},
            {brickMapFuture, false},
            {colorMapFuture, false}
        };
        this->gpuBundle->GetInstance().WaitAny(3, waitInfos, UINT64_MAX);

        // Final raw data:
        const uint32_t* occupancyData = static_cast<const uint32_t*>(this->occupancyReadbackBuffer.GetConstMappedRange(0, sizeof(uint32_t) * 16 * bricksThisPass));
        const BrickOutput* brickOutputData = static_cast<const BrickOutput*>(this->brickOutputReadbackBuffer.GetConstMappedRange(0, sizeof(BrickOutput) * occupiedBrickCount));
        const uint32_t* colorData = static_cast<const uint32_t*>(this->packedColorReadbackBuffer.GetConstMappedRange(0, sizeof(uint32_t) * totalColorCount));

        if (!occupancyData || !brickOutputData || !colorData) 
        {
            std::cerr << "[Voxelizer] Failed to get mapped range from buffers" << std::endl;
            return false;
        }

        for (uint32_t i = 0; i < occupiedBrickCount; i++)
        {
            const BrickOutput& brick = brickOutputData[i];

            uint32_t localBrickIndex = brick.brickGridIndex;
            uint32_t globalBrickIndex = brickStart + localBrickIndex;

            uint32_t occupancy[16];
            std::memcpy(occupancy, &occupancyData[localBrickIndex * 16], sizeof(uint32_t) * 16);

            std::vector<VoxelColorRGB> colors(brick.numOccupied);
            for (uint32_t c = 0; c < brick.numOccupied; c++)
            {
                uint32_t packedColor = colorData[brick.dataOffset + c];
                VoxelColorRGB color;
                colors[c].r = packedColor & 0xFF;
                colors[c].g = (packedColor >> 8) & 0xFF;
                colors[c].b = (packedColor >> 16) & 0xFF;
            }

            VoxelColorRGB lodColor;
            lodColor.r = brick.lodColor & 0xFF;
            lodColor.g = (brick.lodColor >> 8) & 0xFF;
            lodColor.b = (brick.lodColor >> 16) & 0xFF;

            writer.AddBrick(globalBrickIndex, occupancy, colors, lodColor);
        }

        this->occupancyReadbackBuffer.Unmap();
        this->brickOutputReadbackBuffer.Unmap();
        this->packedColorReadbackBuffer.Unmap();

        bricksProcessed += bricksThisPass;

        std::clock_t endPassTime = std::clock();
        double passDuration = double(endPassTime - startPassTime) / CLOCKS_PER_SEC;
        std::cout << "[Voxelizer] Pass " << (pass + 1) << " completed in " 
                  << passDuration << " seconds." << std::endl;
    }

    writer.EndFile();

    std::cout << "[Voxelizer] Voxelization complete. Voxel file saved to " << outputVoxelFile << std::endl;
    return true;
}

// NAIVE IMPLEMENTATION OF VOXELIZATION
/*
//================================//
bool Voxelizer::voxelizeMesh(const std::string& outputVoxelFile, uint32_t voxelResolution)
{
    VoxelFileWriter writer(outputVoxelFile, voxelResolution);

    uint32_t brickResolution = voxelResolution / 8;
    double maxExtent = std::max({meshWidth, meshHeight, meshDepth});
    double voxelSampleSize = maxExtent / static_cast<double>(voxelResolution);

    // Computing of index with x + y*res + z*res^2
    for (uint32_t z = 0; z < brickResolution; z++)
    {
        for (uint32_t y = 0; y < brickResolution; y++)
        {
            for (uint32_t x = 0; x < brickResolution; x++)
            {
                uint32_t brickGridIndex = x + y * brickResolution + z * brickResolution * brickResolution;

                // NAIVE BRICK CHECK AND THEN IF INTERSECTS MESH, FILL OCCUPANCY
                uint32_t occupancy[16] = {0};
                std::vector<VoxelColorRGB> colors;
                uint32_t redSum = 0;
                uint32_t greenSum = 0;
                uint32_t blueSum = 0;

                bool brickOccupied = false;
                size_t numVoxelsInBrick = 0;

                for (uint32_t lz = 0; lz < 8; lz++)
                {
                    for (uint32_t ly = 0; ly < 8; ly++)
                    {
                        for (uint32_t lx = 0; lx < 8; lx++)
                        {
                            // global coords
                            uint32_t gvx = x * 8 + lx;
                            uint32_t gvy = y * 8 + ly;
                            uint32_t gvz = z * 8 + lz;

                            Eigen::Vector3d voxelCenter(
                                meshMinBounds.x() + (gvx + 0.5) * voxelSampleSize, 
                                meshMinBounds.y() + (gvy + 0.5) * voxelSampleSize, 
                                meshMinBounds.z() + (gvz + 0.5) * voxelSampleSize
                            );

                            VoxelColorRGB outColor;
                            if(isVoxelOccupied(voxelCenter, voxelSampleSize, outColor))
                            {
                                brickOccupied = true;
                                
                                uint32_t bitIndex = lx + ly * 8; // on the slice
                                uint32_t sliceIndex = lz * 2; // which slice

                                if (bitIndex < 32)
                                    occupancy[sliceIndex] |= (1u << bitIndex);
                                else
                                    occupancy[sliceIndex + 1] |= (1u << (bitIndex - 32));

                                colors.push_back(outColor);
                                redSum += outColor.r;
                                greenSum += outColor.g;
                                blueSum += outColor.b;

                                numVoxelsInBrick++; 
                            }
                        }
                    }
                }

                if (brickOccupied)
                {
                    VoxelColorRGB lodColor;
                    lodColor.r = static_cast<uint8_t>(redSum / numVoxelsInBrick);
                    lodColor.g = static_cast<uint8_t>(greenSum / numVoxelsInBrick);
                    lodColor.b = static_cast<uint8_t>(blueSum / numVoxelsInBrick);
                    writer.AddBrick(brickGridIndex, occupancy, colors, lodColor);
                }
            }
        }

        if ((z + 1) % (brickResolution / 10 + 1) == 0)
        {
            std::cout << "[Voxelizer] Progress: " 
                      << (100 * (z + 1) / brickResolution) << "%" << std::endl;
        }
    }

    writer.EndFile();
    std::cout << "[Voxelizer] Voxelization complete. Voxel file saved to " << outputVoxelFile << std::endl;
    return true;
}

//================================//
Eigen::Vector3f Voxelizer::sampleTexture(float u, float v) const
{
    if (!hasTexture || !textureData)
        return Eigen::Vector3f(1.0f, 1.0f, 1.0f);
    
    // Clamp UV to [0, 1]
    u = std::clamp(u, 0.0f, 1.0f);
    v = std::clamp(v, 0.0f, 1.0f);
    
    int x = static_cast<int>(u * (texWidth - 1));
    int y = static_cast<int>((1.0f - v) * (texHeight - 1));
    
    int index = (y * texWidth + x) * texChannels;
    
    float r = textureData[index + 0] / 255.0f;
    float g = (texChannels > 1) ? textureData[index + 1] / 255.0f : r;
    float b = (texChannels > 2) ? textureData[index + 2] / 255.0f : r;
    
    return Eigen::Vector3f(r, g, b);
}

//================================//
bool Voxelizer::triangleAABBIntersect(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, const Eigen::Vector3d& v2,
                               const Eigen::Vector3d& boxMin, const Eigen::Vector3d& boxMax)
{
    // Reference: Akenine-Möller "Fast 3D Triangle-Box Overlap Testing"
    // https://fr.scribd.com/document/673258107/Fast-3D-Triangle-Box-Overlap-Testing

    Eigen::Vector3d boxCenter = 0.5 * (boxMin + boxMax);
    Eigen::Vector3d boxHalfSize = 0.5 * (boxMax - boxMin);

    // [1] translate triangle, so the box is centered at the origin 
    Eigen::Vector3d v0t = v0 - boxCenter;
    Eigen::Vector3d v1t = v1 - boxCenter;
    Eigen::Vector3d v2t = v2 - boxCenter;

    Eigen::Vector3d e0 = v1t - v0t;
    Eigen::Vector3d e1 = v2t - v1t;
    Eigen::Vector3d e2 = v0t - v2t;

    // [2] 3 first tests on (1, 0, 0), (0, 1, 0), (0, 0, 1) axes
    Eigen::Vector3d vmin, vmax;
    for (int i = 0; i < 3; ++i)
    {
        vmin[i] = std::min({ v0t[i], v1t[i], v2t[i] });
        vmax[i] = std::max({ v0t[i], v1t[i], v2t[i] });

        if (vmin[i] > boxHalfSize[i] + EPSILON || vmax[i] < -boxHalfSize[i] - EPSILON)
            return false; // No intersection
    }

    // [3] test against the normal of the triangle, "use fast plane/AABB overlap test
    // with the two diagonal vertices whose direction is most closely 
    // aligned with the triangle normal".
    Eigen::Vector3d normal = e0.cross(e1);
    double d = -normal.dot(v0t);
    double r = boxHalfSize.dot(normal.cwiseAbs());
    if (d > r + EPSILON || d < -r - EPSILON)
        return false; // No intersection

    // [4] all 9 other tests
    // aij = (1, 0, 0)i, (0, 1, 0)i, (0, 0, 1)i cross ej
    // ex a00 = (0, -e0.z(), e0.y())
    // we then project all 3 vertices of the triangle onto a00
    // p0 = a00 . v0t = v0.z * v1.y - v0.y * v1.z
    // p1 = a00 . v1t = v0.z * v1.y - v0.y * v1.z = p0
    // p2 = a00 . v2t = (v1.y - v0.y) * v2.z - (v1.z - v0.z) * v2.y
    // then minv = min(p0, p1, p2) = min(p0, p2)
    //      maxv = max(p0, p1, p2) = max(p0, p2)
    // r = boxHalfSize.y() * abs(a00.x) + boxHalfSize.z() * abs(a00.y) = boxHalfSize.y * abs(e0.z) + boxHalfSize.z * abs(e0.y)
    // then check if (minv > r || maxv < -r) -> no intersection
    auto testAxis = [&](const Eigen::Vector3d& axis) -> bool
    {
        double p0 = axis.dot(v0t);
        double p1 = axis.dot(v1t);
        double p2 = axis.dot(v2t);

        double minv = std::min({ p0, p1, p2 });
        double maxv = std::max({ p0, p1, p2 });

        double r = boxHalfSize.x() * std::abs(axis.x()) + boxHalfSize.y() * std::abs(axis.y()) + boxHalfSize.z() * std::abs(axis.z());

        return !(minv > r + EPSILON || maxv < -r - EPSILON);
    };

    if (!testAxis(Eigen::Vector3d(0, -e0.z(), e0.y()))) return false;
    if (!testAxis(Eigen::Vector3d(0, -e1.z(), e1.y()))) return false;
    if (!testAxis(Eigen::Vector3d(0, -e2.z(), e2.y()))) return false;

    if (!testAxis(Eigen::Vector3d(e0.z(), 0, -e0.x()))) return false;
    if (!testAxis(Eigen::Vector3d(e1.z(), 0, -e1.x()))) return false;
    if (!testAxis(Eigen::Vector3d(e2.z(), 0, -e2.x()))) return false;

    if (!testAxis(Eigen::Vector3d(-e0.y(), e0.x(), 0))) return false;
    if (!testAxis(Eigen::Vector3d(-e1.y(), e1.x(), 0))) return false;
    if (!testAxis(Eigen::Vector3d(-e2.y(), e2.x(), 0))) return false;

    return true; // Intersection occurs
}

//================================//
bool Voxelizer::isVoxelOccupied(const Eigen::Vector3d& voxelCenter, double voxelSize, VoxelColorRGB& outColor)
{
    double halfSize = voxelSize * 0.5;
    Eigen::Vector3d voxelMin = voxelCenter.array() - halfSize;
    Eigen::Vector3d voxelMax = voxelCenter.array() + halfSize;
    
    // Check all triangles
    for (int faceIdx = 0; faceIdx < faces.rows(); ++faceIdx)
    {
        Eigen::Vector3d v0 = vertices.row(faces(faceIdx, 0));
        Eigen::Vector3d v1 = vertices.row(faces(faceIdx, 1));
        Eigen::Vector3d v2 = vertices.row(faces(faceIdx, 2));
        
        if (triangleAABBIntersect(v0, v1, v2, voxelMin, voxelMax))
        {
            if (hasTexture && UV.rows() > 0)
            {
                // Sample texture at the barycenter UV
                Eigen::Vector2d uv0 = UV.row(faces(faceIdx, 0));
                Eigen::Vector2d uv1 = UV.row(faces(faceIdx, 1));
                Eigen::Vector2d uv2 = UV.row(faces(faceIdx, 2));
                Eigen::Vector2d uvAvg = (uv0 + uv1 + uv2) / 3.0;
                
                Eigen::Vector3f texColor = sampleTexture(uvAvg.x(), uvAvg.y());
                outColor.r = static_cast<uint8_t>(texColor.x() * 255);
                outColor.g = static_cast<uint8_t>(texColor.y() * 255);
                outColor.b = static_cast<uint8_t>(texColor.z() * 255);
            }
            else
            {
                // Default color based on normal
                Eigen::Vector3d normal = (v1 - v0).cross(v2 - v0).normalized();
                outColor.r = static_cast<uint8_t>((normal.x() * 0.5 + 0.5) * 255);
                outColor.g = static_cast<uint8_t>((normal.y() * 0.5 + 0.5) * 255);
                outColor.b = static_cast<uint8_t>((normal.z() * 0.5 + 0.5) * 255);
            }
            
            return true;
        }
    }
    
    return false;
}
*/