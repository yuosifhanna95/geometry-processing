cmake_minimum_required(VERSION 3.1)
project(assignment5)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"   ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"     ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"    ON)

find_package(LIBIGL REQUIRED QUIET)

add_subdirectory(src/femLib)
add_subdirectory(src/optLib)
include_directories(src/optLib)

FILE(GLOB SOURCES src/test_rosenbrock.cpp)
add_executable(${PROJECT_NAME}_test_rosenbrock ${SOURCES})

FILE(GLOB SRCFILES src/main.cpp src/gui_utils.cpp)
add_executable(${PROJECT_NAME} ${SRCFILES})

target_link_libraries(${PROJECT_NAME} PUBLIC igl::core igl::opengl_glfw igl::opengl_glfw_imgui femLib)
target_link_libraries(${PROJECT_NAME}_test_rosenbrock PUBLIC igl::core igl::opengl_glfw igl::opengl_glfw_imgui femLib)
