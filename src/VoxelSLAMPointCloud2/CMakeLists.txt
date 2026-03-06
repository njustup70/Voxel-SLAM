cmake_minimum_required(VERSION 3.8)
project(voxelslam_pointcloud2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(rviz_rendering REQUIRED)
find_package(rviz_default_plugins REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pluginlib REQUIRED)

set(CMAKE_AUTOMOC ON)

add_library(${PROJECT_NAME} SHARED
  src/voxelslam_pc2.cpp
)

target_link_libraries(${PROJECT_NAME}
  rviz_common::rviz_common
  rviz_rendering::rviz_rendering
  rviz_default_plugins::rviz_default_plugins
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  sensor_msgs
  pluginlib
)

pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

install(TARGETS
  ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(FILES
  plugin_description.xml
  DESTINATION share/${PROJECT_NAME}
)

# Optional: install icons/classes if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/icons")
  install(DIRECTORY icons/
    DESTINATION share/${PROJECT_NAME}/icons)
endif()

ament_package()
