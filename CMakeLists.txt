cmake_minimum_required(VERSION 3.15)

set (CMAKE_CXX_STANDARD 17)

project(VocabTree)

set(SOURCE ./Project/Project/main.cpp)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SOURCE})

# Link against OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
