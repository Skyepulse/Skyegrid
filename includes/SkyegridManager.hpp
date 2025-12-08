#ifndef SKYEGRID_MANAGER_HPP
#define SKYEGRID_MANAGER_HPP

#include "Rendering/RenderEngine.hpp"
#include <GLFW/glfw3.h>

//================================//
class SkyegridManager
{
public:
    SkyegridManager(bool debugMode = false);
    ~SkyegridManager();

    void RunMainLoop();

private:

    std::unique_ptr<RenderEngine> renderEngine;
    std::unique_ptr<WgpuBundle> wgpuBundle;

    std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> window;

    bool correctlyInitialized = false;
    bool debugMode = false;

    RenderInfo renderInfo;
};

#endif