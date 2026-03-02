#ifndef VOXELSLAM_PC2_HPP
#define VOXELSLAM_PC2_HPP

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <rviz_common/message_filter_display.hpp>
#include <rviz_default_plugins/displays/pointcloud/point_cloud_common.hpp>

namespace voxelslam_pointcloud2
{

class PointCloud2Display : public rviz_common::MessageFilterDisplay<sensor_msgs::msg::PointCloud2>
{
  Q_OBJECT
public:
  PointCloud2Display();
  ~PointCloud2Display() override;

  void onInitialize() override;
  void update(float wall_dt, float ros_dt) override;
  void reset() override;

protected:
  void processMessage(sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud) override;

  std::unique_ptr<rviz_default_plugins::PointCloudCommon> point_cloud_common_;
};

} // namespace voxelslam_pointcloud2

#endif
