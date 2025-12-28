#ifndef SKYEGRID_MANAGER_HPP
#define SKYEGRID_MANAGER_HPP

#include "Rendering/RenderEngine.hpp"
#include <GLFW/glfw3.h>

//================================//
class SkyegridManager
{
public:
    SkyegridManager(bool debugMode = false, int voxelResolution = 1168, int maxVisibleBricks = 100000);
    ~SkyegridManager();

    void RunMainLoop();
    
private:
    void ProcessEvents(float deltaTime);
    void UpdateCurrentTime();
    void AccumulateFrameRate();

    std::unique_ptr<RenderEngine> renderEngine;
    std::unique_ptr<WgpuBundle> wgpuBundle;

    std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> window;

    bool correctlyInitialized = false;
    bool debugMode = false;

    RenderInfo renderInfo;

    float lastFrameTime = 0.0f;
    float deltaTime = 0.0f;
    float frameRate = 0.0f;
    std::vector<float> frameRateAccumulator;
};

#endif