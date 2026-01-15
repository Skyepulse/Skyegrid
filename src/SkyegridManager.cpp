#include "../includes/SkyegridManager.hpp"
#include "../includes/constants.hpp"

#include <iostream>
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

const float rotationSpeed = 0.05f;
const float movementSpeed = 0.1f;

//================================//
SkyegridManager::SkyegridManager(bool debugMode, int voxelResolution, int maxVisibleBricks) : debugMode(debugMode), window(nullptr, &glfwDestroyWindow)
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

    WindowFormat windowFormat = { this->window.get(), INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT, false };
    this->renderInfo.width = static_cast<uint32_t>(INITIAL_WINDOW_WIDTH);
    this->renderInfo.height = static_cast<uint32_t>(INITIAL_WINDOW_HEIGHT);
    this->wgpuBundle = std::make_unique<WgpuBundle>(windowFormat);
    this->renderEngine = std::make_unique<RenderEngine>(this->wgpuBundle.get(), voxelResolution, maxVisibleBricks);

    // Initialize Camera position
    this->renderEngine->GetCamera()->SetFov(45.0f);
    float r = static_cast<float>(voxelResolution);

    this->renderEngine->GetCamera()->SetPosition(Eigen::Vector3f(r / 2.0f, r / 2.0f, -r * 1.5f));
    this->renderEngine->GetCamera()->LookAtPoint(Eigen::Vector3f(r / 2.0f, r / 2.0f, r / 2.0f));
    this->renderEngine->GetCamera()->ValidatePixelToRayMatrix();

    this->correctlyInitialized = true;
}

//================================//
SkyegridManager::~SkyegridManager()
{

    this->renderEngine.reset();
    this->wgpuBundle.reset();
    this->window.reset();
    glfwTerminate();
}

//================================//
void SkyegridManager::RunMainLoop()
{
    if (!this->correctlyInitialized)
        return;

    std::cout << "[SkyegridManager] Entering main loop...\n";

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop_arg(
        [](void* arg)
        {
            SkyegridManager* manager = static_cast<SkyegridManager*>(arg);

            manager->UpdateCurrentTime();

            manager->ProcessEvents(manager->deltaTime);

            manager->renderEngine->Render(
                static_cast<void*>(&manager->renderInfo)
            );

            manager->wgpuBundle->GetInstance().ProcessEvents();

            // --- FPS accumulation ---
            manager->AccumulateFrameRate();
        },
        this,
        0,
        true
    );
#else
    while (!glfwWindowShouldClose(this->window.get()))
    {
        this->UpdateCurrentTime();

        this->ProcessEvents(this->deltaTime);
        this->renderEngine->Render(static_cast<void*>(&this->renderInfo));
        //this->renderEngine->RenderDebug(static_cast<void*>(&this->renderInfo));
        this->wgpuBundle->GetSurface().Present();
        this->wgpuBundle->GetInstance().ProcessEvents();

        this->AccumulateFrameRate();
    }
#endif
}

//================================//
void SkyegridManager::ProcessEvents(float deltaTime)
{
    glfwPollEvents();

    Camera* camera = this->renderEngine->GetCamera();

    // Move Speed based on voxel grid size
    float r = static_cast<float>(this->renderEngine->GetVoxelResolution()) * 1.8f;
    float moveSpeed = movementSpeed * r / 100.0f;

    // If shift is held, increase speed
    if (glfwGetKey(this->window.get(), GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
        glfwGetKey(this->window.get(), GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS)
    {
        moveSpeed *= 6.0f;
    }

    Eigen::Vector3f rotationDelta(0.0f, 0.0f, 0.0f);
    if (glfwGetKey(this->window.get(), GLFW_KEY_Q) == GLFW_PRESS)
        rotationDelta.y() -= rotationSpeed * deltaTime * 60.0f;
    if (glfwGetKey(this->window.get(), GLFW_KEY_E) == GLFW_PRESS)
        rotationDelta.y() += rotationSpeed * deltaTime * 60.0f;

    Eigen::Vector3f movementDelta(0.0f, 0.0f, 0.0f);
    if (glfwGetKey(this->window.get(), GLFW_KEY_W) == GLFW_PRESS)
        movementDelta.z() += moveSpeed * deltaTime * 60.0f;
    if (glfwGetKey(this->window.get(), GLFW_KEY_S) == GLFW_PRESS)
        movementDelta.z() -= moveSpeed * deltaTime * 60.0f;
    if (glfwGetKey(this->window.get(), GLFW_KEY_A) == GLFW_PRESS)
        movementDelta.x() -= moveSpeed * deltaTime * 60.0f;
    if (glfwGetKey(this->window.get(), GLFW_KEY_D) == GLFW_PRESS)
        movementDelta.x() += moveSpeed * deltaTime * 60.0f;
    //Z, x for up and down
    if (glfwGetKey(this->window.get(), GLFW_KEY_Z) == GLFW_PRESS)
        movementDelta.y() += moveSpeed * deltaTime * 60.0f;
    if (glfwGetKey(this->window.get(), GLFW_KEY_X) == GLFW_PRESS)
        movementDelta.y() -= moveSpeed * deltaTime * 60.0f;
    
    camera->Rotate(rotationDelta);
    camera->Move(movementDelta);

    if (glfwGetKey(this->window.get(), GLFW_KEY_R) == GLFW_PRESS)
        std::cout << "[SkyegridManager] Camera Position: " << camera->GetPosition().transpose() << std::endl;

    WindowFormat currentFormat = this->wgpuBundle->GetWindowFormat();
    this->renderInfo.width = static_cast<uint32_t>(currentFormat.width);
    this->renderInfo.height = static_cast<uint32_t>(currentFormat.height);
    this->renderInfo.resizeNeeded = currentFormat.resizeNeeded;
}

//================================//
void SkyegridManager::UpdateCurrentTime()
{
#ifdef __EMSCRIPTEN__
    double currentTime = emscripten_get_now() / 1000.0;
#else
    double currentTime = static_cast<double>(glfwGetTime());
#endif

    if (this->lastFrameTime == 0.0f)
            this->lastFrameTime = static_cast<float>(currentTime);

    this->deltaTime = static_cast<float>(currentTime - this->lastFrameTime);
    this->lastFrameTime = static_cast<float>(currentTime);

    this->renderInfo.time = currentTime;

#ifdef __EMSCRIPTEN__
    this->deltaTime = std::min(this->deltaTime, 0.1f);
#endif
}

//================================//
void SkyegridManager::AccumulateFrameRate()
{
    if (this->deltaTime <= 0.0f)
        return;

    this->frameRateAccumulator.push_back(1.0f / this->deltaTime);
    if (this->frameRateAccumulator.size() >= 100)
    {
        float sum = 0.0f;
        for (float fr : this->frameRateAccumulator)
            sum += fr;
        this->frameRate = sum / static_cast<float>(this->frameRateAccumulator.size());
        this->frameRateAccumulator.clear();

        std::cout << "[SkyegridManager] Average Frame Rate: " << this->frameRate << " FPS" << std::endl;
    }
}