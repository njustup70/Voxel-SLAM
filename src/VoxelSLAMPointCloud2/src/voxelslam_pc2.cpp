#include <memory>
#include <vector>
#include <string>

#include <OgreSceneNode.h>
#include <OgreSceneManager.h>

#include <rviz_common/display_context.hpp>
#include <rviz_common/frame_manager_iface.hpp>
#include <rviz_common/validate_floats.hpp>
#include <rviz_default_plugins/displays/pointcloud/point_cloud_common.hpp>

#include "voxelslam_pc2.hpp"

namespace voxelslam_pointcloud2
{

PointCloud2Display::PointCloud2Display()
  : point_cloud_common_(std::make_unique<rviz_default_plugins::PointCloudCommon>(this))
{
}

PointCloud2Display::~PointCloud2Display() = default;

void PointCloud2Display::onInitialize()
{
  // Initialize the message filter display with a specific QoS for Best Effort
  // By default, rviz_common::MessageFilterDisplay uses the QoS provided by the display properties.
  // We call MFDClass::onInitialize() which sets up the subscription.
  MFDClass::onInitialize();
  point_cloud_common_->initialize(context_, scene_node_);
}

void PointCloud2Display::processMessage(sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud)
{
  auto filtered = std::make_shared<sensor_msgs::msg::PointCloud2>();
  
  int xi = -1, yi = -1, zi = -1;
  for (size_t i = 0; i < cloud->fields.size(); ++i) {
    if (cloud->fields[i].name == "x") xi = (int)i;
    else if (cloud->fields[i].name == "y") yi = (int)i;
    else if (cloud->fields[i].name == "z") zi = (int)i;
  }

  if (xi == -1 || yi == -1 || zi == -1) {
    return;
  }

  const uint32_t xoff = cloud->fields[xi].offset;
  const uint32_t yoff = cloud->fields[yi].offset;
  const uint32_t zoff = cloud->fields[zi].offset;
  const uint32_t point_step = cloud->point_step;
  const size_t point_count = cloud->width * cloud->height;

  if (point_count * point_step != cloud->data.size()) {
    return;
  }

  filtered->data.resize(cloud->data.size());
  uint32_t output_count = 0;
  if (point_count > 0) {
    uint8_t* output_ptr = filtered->data.data();
    const uint8_t* ptr = cloud->data.data();
    const uint8_t* ptr_end = ptr + cloud->data.size();
    
    for (; ptr < ptr_end; ptr += point_step) {
      float x = *reinterpret_cast<const float*>(ptr + xoff);
      float y = *reinterpret_cast<const float*>(ptr + yoff);
      float z = *reinterpret_cast<const float*>(ptr + zoff);
      
      if (rviz_common::validateFloats(x) && rviz_common::validateFloats(y) && rviz_common::validateFloats(z)) {
        memcpy(output_ptr, ptr, point_step);
        output_ptr += point_step;
        output_count++;
      }
    }
  }

  filtered->header = cloud->header;
  filtered->fields = cloud->fields;
  filtered->data.resize(output_count * point_step);
  filtered->height = 1;
  filtered->width = output_count;
  filtered->is_bigendian = cloud->is_bigendian;
  filtered->point_step = point_step;
  filtered->row_step = output_count * point_step;
  filtered->is_dense = true;

  if (output_count > 0) {
    point_cloud_common_->addMessage(filtered);
  } else {
    point_cloud_common_->reset();
  }
}

void PointCloud2Display::update(float wall_dt, float ros_dt)
{
  point_cloud_common_->update(wall_dt, ros_dt);
}

void PointCloud2Display::reset()
{
  MFDClass::reset();
  point_cloud_common_->reset();
}

} // namespace voxelslam_pointcloud2

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(voxelslam_pointcloud2::PointCloud2Display, rviz_common::Display)
