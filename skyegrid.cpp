#include <iostream>
#include "includes/SkyegridManager.hpp"
#include <charconv>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

const int voxel_resolution =  1024;
const int max_visible_bricks =  1;

std::unique_ptr<SkyegridManager> g_manager;

//================================//
int main(int argc, char** argv)
{
    // parse first argument as voxel resolution if present
    int value1 = voxel_resolution;
    if (argc > 1)
    {
        // throw error if not a valid integer
        const char* str = argv[1];
        auto result = std::from_chars(str, str + std::strlen(str), value1);

        if (result.ec != std::errc()) { // Parsing failed
            std::cerr << "Error: Invalid integer: '" << str << "'\n";
            value1 = voxel_resolution;
        }
        if (result.ptr != str + std::strlen(str)) { // Trailing characters
            std::cerr << "Error: Trailing characters after number: '" << str << "'\n";
            value1 = voxel_resolution;
        }
    }

    int value2 = max_visible_bricks;
    if(argc > 2)
    {
        const char* str = argv[2];
        auto result = std::from_chars(str, str + std::strlen(str), value2);
        if (result.ec != std::errc()) { // Parsing failed
            std::cerr << "Error: Invalid integer: '" << str << "'\n";
            value2 = max_visible_bricks;
        }
        if (result.ptr != str + std::strlen(str)) { // Trailing characters
            std::cerr << "Error: Trailing characters after number: '" << str << "'\n";
            value2 = max_visible_bricks;
        }
    }

    std::string fileName = "data/ov.vox";
    if (argc > 3)
    {
        fileName = argv[3];
    }

    g_manager = std::make_unique<SkyegridManager>(false, value1, value2);
    g_manager->LoadVoxelFile(fileName);
    g_manager->InitGraphics();
    g_manager->RunMainLoop();

#ifndef __EMSCRIPTEN__
    g_manager.reset();
#endif

    std::cout << "Exiting application.\n";
    return 0;
}