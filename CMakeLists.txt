cmake_minimum_required(VERSION 4.0)
project(Skyegrid)
set(CMAKE_CXX_STANDARD 20)

# Old policy to fetch IGL's Eigen
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

set(SRC_FILES
  main.cpp
)

add_executable(Skyegrid ${SRC_FILES})

include(FetchContent)

# IGL
FetchContent_Declare(
  libigl
  GIT_REPOSITORY https://github.com/libigl/libigl.git
  GIT_TAG        v2.6.0
)

FetchContent_MakeAvailable(libigl)

# Dawn specifics
set(DAWN_ROOT "${CMAKE_SOURCE_DIR}/dawn")
set(DAWN_FETCH_DEPENDENCIES ON CACHE BOOL "" FORCE) # Fetch all dawn dependencies
set(DAWN_USE_BUILT_DXC ON CACHE BOOL "" FORCE)

add_subdirectory("${DAWN_ROOT}" "${CMAKE_BINARY_DIR}/dawn_build" EXCLUDE_FROM_ALL) # Exclude from all in order to not build dawn by default

if(EMSCRIPTEN)
  set(DCMAKE_CXX_SCAN_FOR_MODULES OFF CACHE BOOL "" FORCE) # Disable C++ modules for Emscripten
  set_target_properties(Skyegrid PROPERTIES SUFFIX ".html")
  target_link_libraries(Skyegrid PRIVATE emdawnwebgpu_cpp webgpu_glfw igl::core)
  target_link_options(Skyegrid PRIVATE "-sASYNCIFY=1" "-sUSE_GLFW=3")
else()
    target_link_libraries(
            Skyegrid
            PRIVATE
        webgpu_dawn
        webgpu_glfw
        glfw
        igl::core
    )
endif()

# On Linux they are sometimes required
find_package(Threads REQUIRED)
target_link_libraries(Skyegrid PRIVATE ${OPENGL_gl_LIBRARY} Threads::Threads)