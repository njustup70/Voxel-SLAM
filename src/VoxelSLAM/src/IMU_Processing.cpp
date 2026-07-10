#include "parameters.h"
#include "IMU_Processing.h"

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true)
{
  imu_en = true;
  init_iter_num = 1;
  mean_acc = V3D(0, 0, 0.0);
  after_imu_init_ = false;
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset()
{
  RCLCPP_WARN(rclcpp::get_logger("ImuProcess"), "Reset ImuProcess");
  mean_acc = V3D(0, 0, 0.0);
  imu_need_init_ = true;
  init_iter_num = 1;
  after_imu_init_ = false;
  b_first_frame_ = true;
}

void ImuProcess::Set_init(Eigen::Vector3d &tmp_gravity, Eigen::Matrix3d &rot)
{
  V3D grav(gravity[0], gravity[1], gravity[2]);
  M3D hat_grav;
  hat_grav << 0.0, grav(2), -grav(1),
              -grav(2), 0.0, grav(0),
              grav(1), -grav(0), 0.0;
  double align_norm = (hat_grav * tmp_gravity).norm() / grav.norm() / tmp_gravity.norm();
  double align_cos = grav.transpose() * tmp_gravity;
  align_cos = align_cos / grav.norm() / tmp_gravity.norm();
  if (align_norm < 1e-6)
  {
    rot = (align_cos > 1e-6) ? Eye3d : -Eye3d;
  }
  else
  {
    V3D align_angle = hat_grav * tmp_gravity / (hat_grav * tmp_gravity).norm() * acos(align_cos);
    rot = Exp(align_angle(0), align_angle(1), align_angle(2));
  }
}

void ImuProcess::IMU_init(const MeasureGroup &meas, int &N)
{
  RCLCPP_INFO(rclcpp::get_logger("ImuProcess"), "IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
  V3D cur_acc;
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
  }
  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_acc += (cur_acc - mean_acc) / N;
    N++;
  }
}

void ImuProcess::Process(const MeasureGroup &meas, PointCloudXYZI::Ptr cur_pcl_un_)
{
  if (\!imu_en || meas.imu.empty())
  {
    *cur_pcl_un_ = *(meas.lidar);
    return;
  }
  if (imu_need_init_)
  {
    IMU_init(meas, init_iter_num);
    if (init_iter_num > MAX_INI_COUNT)
    {
      RCLCPP_INFO(rclcpp::get_logger("ImuProcess"), "IMU init done");
      imu_need_init_ = false;
      *cur_pcl_un_ = *(meas.lidar);
    }
    return;
  }
  if (\!after_imu_init_) after_imu_init_ = true;
  *cur_pcl_un_ = *(meas.lidar);
}
