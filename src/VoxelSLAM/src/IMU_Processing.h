#pragma once
#include <cmath>
#include <deque>
#include <Eigen/Eigen>
#include "common_lib.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#define MAX_INI_COUNT (100)

class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ImuProcess();
  ~ImuProcess();
  void Reset();
  void Process(const MeasureGroup &meas, PointCloudXYZI::Ptr cur_pcl_un_);
  void Set_init(Eigen::Vector3d &tmp_gravity, Eigen::Matrix3d &rot);

  bool imu_en = true;
  V3D    mean_acc = V3D::Zero();
  bool   imu_need_init_ = true;
  bool   after_imu_init_ = false;
  bool   b_first_frame_ = true;
  int    init_iter_num = 1;

 private:
  void IMU_init(const MeasureGroup &meas, int &N);
  V3D cov_gyr_scale_ = V3D::Zero();
  V3D cov_vel_scale_ = V3D::Zero();
};
