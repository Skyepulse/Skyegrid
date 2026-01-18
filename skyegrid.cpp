#include <iostream>
#include "includes/SkyegridManager.hpp"
#include <charconv>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

const int voxel_resolution = 1024;
const int max_visible_bricks = 100000;

#ifdef __EMSCRIPTEN__
std::unique_ptr<SkyegridManager> g_manager;
#endif

int main(int argc, char** argv)
{
    int value1 = voxel_resolution;
    int value2 = max_visible_bricks;
    
#ifdef __EMSCRIPTEN__
    std::string fileName = "/data/ov.vox";
#else
    if (argc > 1)
    {
        const char* str = argv[1];
        auto result = std::from_chars(str, str + std::strlen(str), value1);
        if (result.ec != std::errc() || result.ptr != str + std::strlen(str)) {
            std::cerr << "Error: Invalid integer: '" << str << "'\n";
            value1 = voxel_resolution;
        }
    }
    
    if (argc > 2)
    {
        const char* str = argv[2];
        auto result = std::from_chars(str, str + std::strlen(str), value2);
        if (result.ec != std::errc() || result.ptr != str + std::strlen(str)) {
            std::cerr << "Error: Invalid integer: '" << str << "'\n";
            value2 = max_visible_bricks;
        }
    }
    
    std::string fileName = "data/ov.vox";
    if (argc > 3)
    {
        fileName = argv[3];
    }
#endif

#ifdef __EMSCRIPTEN__
    g_manager = std::make_unique<SkyegridManager>(false, value1, value2);
    g_manager->LoadVoxelFile(fileName);
    g_manager->InitGraphics();
    g_manager->RunMainLoop();
#else
    std::unique_ptr<SkyegridManager> manager = std::make_unique<SkyegridManager>(false, value1, value2);
    manager->LoadVoxelFile(fileName);
    manager->InitGraphics();
    manager->RunMainLoop();
    manager.reset();
    std::cout << "Exiting application.\n";
#endif

    return 0;
}