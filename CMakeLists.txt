cmake_minimum_required(VERSION 4.0)
project(Skyegrid)
set(CMAKE_CXX_STANDARD 20)

# Old policy to fetch IGL's Eigen
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

set(SRC_FILES
  skyegrid.cpp
  includes/Rendering/wgpuBundle.hpp
  src/Rendering/wgpuBundle.cpp
  includes/Rendering/wgpuHelpers.hpp
  src/Rendering/wgpuHelpers.cpp
  includes/Rendering/Pipelines/pipelines.hpp
  src/Rendering/Pipelines/pipelines.cpp
  src/SkyegridManager.cpp
  includes/SkyegridManager.hpp
  src/Rendering/RenderEngine.cpp
  includes/Rendering/RenderEngine.hpp
  includes/constants.hpp
  includes/Rendering/Camera/Camera.hpp
  src/Rendering/Camera/Camera.cpp
  src/VoxelManager.cpp
  includes/VoxelManager.hpp
  includes/Voxelizer.hpp
  src/Voxelizer.cpp
)

set(VOXELIZER_SRC_FILES
  main_voxelize.cpp
  includes/Voxelizer.hpp
  includes/Rendering/wgpuHelpers.hpp
  src/Rendering/wgpuHelpers.cpp
  src/Voxelizer.cpp
  src/Rendering/wgpuBundle.cpp
  includes/Rendering/wgpuBundle.hpp
  includes/Rendering/Pipelines/pipelines.hpp
  src/Rendering/Pipelines/pipelines.cpp
)

add_executable(Skyegrid ${SRC_FILES})
add_executable(Voxelizer ${VOXELIZER_SRC_FILES})

include(FetchContent)

# IGL
FetchContent_Declare(
  libigl
  GIT_REPOSITORY https://github.com/libigl/libigl.git
  GIT_TAG        v2.6.0
)
FetchContent_MakeAvailable(libigl)

# ImGUI
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.92.5
)
FetchContent_MakeAvailable(imgui)

# SKYEGRID CONFIGURATION

set(IMGUI_DIR ${imgui_SOURCE_DIR})
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
)
set(IMGUI_BACKEND_SOURCES
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_wgpu.cpp
)
target_sources(Skyegrid PRIVATE 
    ${IMGUI_SOURCES} 
    ${IMGUI_BACKEND_SOURCES}
)
target_include_directories(Skyegrid PRIVATE 
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CMAKE_SOURCE_DIR}/includes
)

# VOXELIZER CONFIGURATION
target_include_directories(Voxelizer PRIVATE 
    ${CMAKE_SOURCE_DIR}/includes
    ${stb_SOURCE_DIR}  # For stb_image.h
)
target_link_libraries(Voxelizer PRIVATE igl::core)

# DAWN SPECIFICS

set(DAWN_ROOT "${CMAKE_SOURCE_DIR}/dawn")
set(DAWN_FETCH_DEPENDENCIES ON CACHE BOOL "" FORCE) # Fetch all dawn dependencies
set(DAWN_USE_BUILT_DXC ON CACHE BOOL "" FORCE)

add_subdirectory("${DAWN_ROOT}" "${CMAKE_BINARY_DIR}/dawn_build" EXCLUDE_FROM_ALL) # Exclude from all in order to not build dawn by default

if(EMSCRIPTEN)

  target_compile_definitions(Skyegrid PRIVATE 
    IMGUI_IMPL_WEBGPU_BACKEND_EMSCRIPTEN
  )
  target_compile_definitions(Voxelizer PRIVATE
    IMGUI_IMPL_WEBGPU_BACKEND_EMSCRIPTEN
  )

  set(DCMAKE_CXX_SCAN_FOR_MODULES OFF CACHE BOOL "" FORCE) # Disable C++ modules for Emscripten
  set_target_properties(Skyegrid PROPERTIES SUFFIX ".html")
  target_link_libraries(Skyegrid PRIVATE emdawnwebgpu_cpp webgpu_glfw igl::core)
  target_link_options(Skyegrid PRIVATE "-sINITIAL_MEMORY=64MB" "-sALLOW_MEMORY_GROWTH=1" "-sASYNCIFY=1" "-sUSE_GLFW=3" "--preload-file=${CMAKE_SOURCE_DIR}/Shaders@/Shaders")

  set_target_properties(Voxelizer PROPERTIES SUFFIX ".html")
  target_link_libraries(Voxelizer PRIVATE emdawnwebgpu_cpp webgpu_glfw igl::core)
  target_link_options(Voxelizer PRIVATE "-sINITIAL_MEMORY=64MB" "-sALLOW_MEMORY_GROWTH=1" "-sASYNCIFY=1" "-sUSE_GLFW=3" "--preload-file=${CMAKE_SOURCE_DIR}/Shaders@/Shaders")

else()

  target_compile_definitions(Skyegrid PRIVATE 
    IMGUI_IMPL_WEBGPU_BACKEND_DAWN
  )
  target_compile_definitions(Voxelizer PRIVATE
    IMGUI_IMPL_WEBGPU_BACKEND_DAWN
  )

  target_link_libraries(
          Skyegrid
          PRIVATE
      webgpu_dawn
      webgpu_glfw
      glfw
      igl::core
  )

  target_link_libraries(
          Voxelizer
          PRIVATE
      webgpu_dawn
      webgpu_glfw
      glfw
      igl::core
  )
  
  add_custom_command(TARGET Skyegrid POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_SOURCE_DIR}/Shaders
      $<TARGET_FILE_DIR:Skyegrid>/Shaders
  )

endif()

if(NOT EMSCRIPTEN)
  find_package(Threads REQUIRED)
  target_link_libraries(Skyegrid PRIVATE Threads::Threads)
endif()