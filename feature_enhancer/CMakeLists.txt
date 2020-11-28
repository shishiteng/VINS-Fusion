cmake_minimum_required(VERSION 2.8.3)
project(feature_enhancer)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    pcl_ros
    pcl_conversions
)

find_package(OpenCV 3 REQUIRED)

catkin_package()

include_directories(
    ${catkin_INCLUDE_DIRS}
)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(feature_enhancer_lidar
    src/feature_enhancer_lidar.cpp
)
target_link_libraries(feature_enhancer_lidar ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(feature_enhancer_rgbd
    src/feature_enhancer_rgbd.cpp
)
target_link_libraries(feature_enhancer_rgbd ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(depth_to_camera
    src/depth_to_camera.cpp
)
target_link_libraries(depth_to_camera ${catkin_LIBRARIES} ${OpenCV_LIBS})
