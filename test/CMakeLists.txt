cmake_minimum_required(VERSION 4.0)
project(Skyegrid)
set(CMAKE_CXX_STANDARD 20)

add_executable(Skyegrid main.cpp)

# Dawn specifics
set(DAWN_ROOT "${CMAKE_SOURCE_DIR}/../dawn")
set(DAWN_FETCH_DEPENDENCIES ON CACHE BOOL "" FORCE) # Fetch all dawn dependencies
set(DAWN_USE_BUILT_DXC ON CACHE BOOL "" FORCE)

add_subdirectory("${DAWN_ROOT}" "${CMAKE_BINARY_DIR}/dawn_build" EXCLUDE_FROM_ALL) # Exclude from all in order to not build dawn by default



if(EMSCRIPTEN)
  set(DCMAKE_CXX_SCAN_FOR_MODULES OFF CACHE BOOL "" FORCE) # Disable C++ modules for Emscripten
  set_target_properties(Skyegrid PROPERTIES SUFFIX ".html")
  target_link_libraries(Skyegrid PRIVATE emdawnwebgpu_cpp webgpu_glfw)
  target_link_options(Skyegrid PRIVATE "-sASYNCIFY=1" "-sUSE_GLFW=3")
else()
    target_link_libraries(
            Skyegrid
            PRIVATE
        webgpu_dawn
        webgpu_glfw
        glfw
    )
endif()