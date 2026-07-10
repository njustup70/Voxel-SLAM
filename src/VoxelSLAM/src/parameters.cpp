#include "parameters.h"
#include "IMU_Processing.h"

// IKFoM globals
esekfom::esekf<state_input, 24, input_ikfom> kf_input;
esekfom::esekf<state_output, 30, input_ikfom> kf_output;
input_ikfom input_in;

// Parameters
bool prop_at_freq_of_imu = true;
bool use_imu_as_input = false, space_down_sample = true;
bool publish_odometry_without_downsample = true;
bool extrinsic_est_en = false;
int  init_map_size = 10;
double imu_time_inte = 0.005;
double laser_point_cov = 0.01;
double acc_cov_input = 0.1, gyr_cov_input = 0.1, vel_cov = 20;
double gyr_cov_output = 0.1, acc_cov_output = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0.5, filter_size_map_min = 0.5;
bool imu_en = true;
std::vector<double> gravity;
double G_m_s2 = 9.81;
double time_update_last = 0.0, time_current = 0.0, time_predict_last_const = 0.0;

// Extrinsic
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);
std::vector<double> extrinT(3, 0.0);
std::vector<double> extrinR(9, 0.0);

std::shared_ptr<ImuProcess> p_imu;
MeasureGroup Measures;

void readParameters(rclcpp::Node::SharedPtr node)
{
  p_imu.reset(new ImuProcess());

  node->declare_parameter<bool>("prop_at_freq_of_imu", true);
  node->declare_parameter<bool>("use_imu_as_input", false);
  node->declare_parameter<bool>("space_down_sample", true);
  node->declare_parameter<bool>("odometry.publish_odometry_without_downsample", true);
  node->declare_parameter<bool>("mapping.extrinsic_est_en", false);
  node->declare_parameter<bool>("mapping.imu_en", true);
  node->declare_parameter<int>("init_map_size", 10);

  node->declare_parameter<double>("mapping.acc_cov_input", 0.1);
  node->declare_parameter<double>("mapping.gyr_cov_input", 0.1);
  node->declare_parameter<double>("mapping.vel_cov", 20.0);
  node->declare_parameter<double>("mapping.gyr_cov_output", 0.1);
  node->declare_parameter<double>("mapping.acc_cov_output", 0.1);
  node->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
  node->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
  node->declare_parameter<double>("mapping.lidar_meas_cov", 0.01);
  node->declare_parameter<double>("mapping.imu_time_inte", 0.005);
  node->declare_parameter<double>("filter_size_surf", 0.5);
  node->declare_parameter<double>("filter_size_map", 0.5);

  node->declare_parameter<std::vector<double>>("mapping.gravity", {0.0, 0.0, -9.81});
  node->declare_parameter<std::vector<double>>("mapping.extrinsic_T", {0.0, 0.0, 0.0});
  node->declare_parameter<std::vector<double>>("mapping.extrinsic_R", {1,0,0,0,1,0,0,0,1});

  node->get_parameter<bool>("prop_at_freq_of_imu", prop_at_freq_of_imu);
  node->get_parameter<bool>("use_imu_as_input", use_imu_as_input);
  node->get_parameter<bool>("space_down_sample", space_down_sample);
  node->get_parameter<bool>("odometry.publish_odometry_without_downsample", publish_odometry_without_downsample);
  node->get_parameter<bool>("mapping.extrinsic_est_en", extrinsic_est_en);
  node->get_parameter<bool>("mapping.imu_en", imu_en);
  node->get_parameter<int>("init_map_size", init_map_size);

  node->get_parameter<double>("mapping.acc_cov_input", acc_cov_input);
  node->get_parameter<double>("mapping.gyr_cov_input", gyr_cov_input);
  node->get_parameter<double>("mapping.vel_cov", vel_cov);
  node->get_parameter<double>("mapping.gyr_cov_output", gyr_cov_output);
  node->get_parameter<double>("mapping.acc_cov_output", acc_cov_output);
  node->get_parameter<double>("mapping.b_gyr_cov", b_gyr_cov);
  node->get_parameter<double>("mapping.b_acc_cov", b_acc_cov);
  node->get_parameter<double>("mapping.lidar_meas_cov", laser_point_cov);
  node->get_parameter<double>("mapping.imu_time_inte", imu_time_inte);
  node->get_parameter<double>("filter_size_surf", filter_size_surf_min);
  node->get_parameter<double>("filter_size_map", filter_size_map_min);

  node->get_parameter<std::vector<double>>("mapping.gravity", gravity);
  node->get_parameter<std::vector<double>>("mapping.extrinsic_T", extrinT);
  node->get_parameter<std::vector<double>>("mapping.extrinsic_R", extrinR);

  if (gravity.empty()) gravity = {0.0, 0.0, -9.81};
  G_m_s2 = std::sqrt(gravity[0]*gravity[0] + gravity[1]*gravity[1] + gravity[2]*gravity[2]);

  Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
  Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);

  p_imu->imu_en = imu_en;
}
