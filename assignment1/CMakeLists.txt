cmake_minimum_required(VERSION 3.1)
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME})

### conditionally compile certain modules depending on libraries found on the system    
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../libigl/cmake)   
  
option(LIBIGL_WITH_OPENGL            "Use OpenGL"                   ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"                     ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"                    ON)

### Adding libIGL: choose the path to your local copy libIGL    
if(NOT TARGET igl_common)   
  include(libigl)   
else()    
  include(LibiglDownloadExternal)   
endif()

### Output directories
if(MSVC)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
else()
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../")
endif()

add_executable(${PROJECT_NAME}_bin ${PROJECT_SOURCE_DIR}/src/main.cpp ${PROJECT_SOURCE_DIR}/src/gui_utils.cpp)
target_link_libraries(${PROJECT_NAME}_bin igl::core igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui)