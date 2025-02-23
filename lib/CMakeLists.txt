# External dependencies
include(FetchContent)

FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb
    GIT_TAG master
    GIT_SHALLOW ON
)
if(NOT stb_POPULATED)
    FetchContent_Populate(stb)
endif()

if(WIN32)
    set(VOLK_STATIC_DEFINES "VK_USE_PLATFORM_WIN32_KHR" CACHE STRING "")
else()
    set(VOLK_STATIC_DEFINES "VK_USE_PLATFORM_XLIB_KHR" CACHE STRING "")
endif()

FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk
    GIT_TAG 1.3.250
    GIT_SHALLOW ON
)
FetchContent_MakeAvailable(volk)

set(TINYEXR_BUILD_SAMPLE OFF CACHE BOOL "")
FetchContent_Declare(
    tinyexr
    GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
    GIT_TAG v1.0.1
    GIT_SHALLOW ON
)
FetchContent_MakeAvailable(tinyexr)

set(VMA_STATIC_VULKAN_FUNCTIONS OFF CACHE BOOL "")
set(VMA_DYNAMIC_VULKAN_FUNCTIONS ON CACHE BOOL "")
FetchContent_Declare(
    vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
    GIT_TAG v3.0.0
    GIT_SHALLOW ON
)
if(NOT vma_POPULATED)
    FetchContent_Populate(vma)
endif()

add_library(
    lflop
    Buffer.cpp
    Buffer.hpp
    ColorMaps.cpp
    ColorMaps.hpp
    Flop.cpp
    FlopContext.hpp
    Fullscreen.cpp
    Fullscreen.hpp
    Image.cpp
    Image.hpp
    Kernel.cpp
    Kernel.hpp
    STB.cpp
    VkGlobals.hpp
    VMA.cpp
)

target_link_libraries(lflop PUBLIC flop_shaders tinyexr)

target_compile_features(
    lflop
    PRIVATE
    cxx_std_20
)

target_include_directories(
    lflop
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${vma_SOURCE_DIR}/include
    PRIVATE
    ${stb_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}
)

target_link_libraries(
    lflop
    PRIVATE
    volk
)
