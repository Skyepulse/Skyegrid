#include "includes/Voxelizer.hpp"
#include <iostream>
#include <string>
#include <charconv>
#include <cstring>

//================================//
int main(int argc, char** argv)
{
    // parse first arg as input mesh file, second arg as output voxel file, third arg as voxel resolution
    std::string inputMeshFile = "meshes/wallE.ply";
    std::string outputVoxelFile = "data/output_voxel.vox";
    uint32_t voxelResolution = 16;

    if (argc > 1)
    {
        inputMeshFile = argv[1];
    }
    else
    {
        std::cerr << "Error: No input mesh file provided.\n";
        return 1;
    }

    if (argc > 2)
    {
        outputVoxelFile = argv[2];
    }

    if (argc > 3)
    {
        const char* str = argv[3];
        auto result = std::from_chars(str, str + std::strlen(str), voxelResolution);

        if (result.ec != std::errc()) { // Parsing failed
            std::cerr << "Error: Invalid voxel resolution: '" << str << "'. Using default 128.\n";
            voxelResolution = 128;
        }
        if (result.ptr != str + std::strlen(str)) { // Trailing characters
            std::cerr << "Error: Trailing characters after number: '" << str << "'. Using default 128.\n";
            voxelResolution = 128;
        }
    }

    Voxelizer voxelizer = Voxelizer();
    if (!voxelizer.loadMesh(inputMeshFile))
    {
        std::cerr << "Error: Failed to load mesh from file: " << inputMeshFile << "\n";
        return 1;
    }

    uint32_t maxBricksPerPass;
    uint8_t numPasses;
    voxelizer.checkLimits(voxelResolution, maxBricksPerPass, numPasses);
    if (!voxelizer.voxelizeMesh(outputVoxelFile, voxelResolution, maxBricksPerPass, numPasses))
    {
        std::cerr << "Error: Failed to voxelize mesh and save to file: " << outputVoxelFile << "\n";
        return 1;
    }

    return 0;
}