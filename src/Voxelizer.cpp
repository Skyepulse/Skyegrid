#include "../../includes/Voxelizer.hpp"

#include <igl/readPLY.h>
#include <iostream>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" 

//================================//
Voxelizer::Voxelizer()
{
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
    // return false if not .ply

    bool success = igl::readPLY(filename, this->vertices, this->faces, this->edges, this->Normals, this->UV);
    if (success)
    {
        std::cout << "[Voxelizer] Successfully loaded mesh from " << filename << " with "
                  << this->vertices.rows() << " vertices and "
                  << this->faces.rows() << " faces" << std::endl;
    }

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
        this->hasTexture = safeTextureLoad(defaultTexturePath, &this->textureData, &this->texWidth, &this->texHeight, &this->texChannels);
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