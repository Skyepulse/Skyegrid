#include "../includes/SkyegridManager.hpp"
#include "../includes/constants.hpp"

#include <iostream>
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

//================================//
SkyegridManager::SkyegridManager(bool debugMode) : debugMode(debugMode), window(nullptr, &glfwDestroyWindow)
{
    if (!glfwInit())
    {
        std::cout << "[SkyeGridManager] Failed to initialize GLFW" << std::endl;
        return;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    this->window.reset(glfwCreateWindow(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT, "Skyegrid", nullptr, nullptr));

     GLFWwindow* window = this->window.get();

    if (!window)
    {
        std::cerr << "[SkyeGridManager] Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }

    WindowFormat windowFormat = { this->window.get(), INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT };
    this->wgpuBundle = std::make_unique<WgpuBundle>(windowFormat);
    this->renderEngine = std::make_unique<RenderEngine>(this->wgpuBundle.get());

    this->correctlyInitialized = true;
}

//================================//
SkyegridManager::~SkyegridManager()
{
    glfwDestroyWindow(this->window.get());
    glfwTerminate();
}

//================================//
void SkyegridManager::RunMainLoop()
{
    if (!this->correctlyInitialized)
        return;

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop_arg(
        [](void* arg)
        {
            SkyegridManager* manager = static_cast<SkyegridManager*>(arg);
            manager->renderEngine->Render(static_cast<void*>(manager->wgpuBundle.get()));
        },
        this,
        0,
        true
    );
#else
    while (!glfwWindowShouldClose(this->window.get()))
    {
        glfwPollEvents();
        this->renderEngine->Render(static_cast<void*>(this->wgpuBundle.get()));
        this->wgpuBundle->GetSurface().Present();
        this->wgpuBundle->GetInstance().ProcessEvents();
    }
#endif
}