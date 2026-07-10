#pragma once
#include "common_lib.h"
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>
#include <memory>

class ImuProcess;

// --- IKFoM global instances (defined in voxelslam.cpp) ---
extern esekfom::esekf<state_input, 24, input_ikfom> kf_input;
extern esekfom::esekf<state_output, 30, input_ikfom> kf_output;
extern input_ikfom input_in;

// --- Point-LIO style parameters ---
extern bool prop_at_freq_of_imu;
extern bool use_imu_as_input, space_down_sample;
extern bool publish_odometry_without_downsample;
extern bool extrinsic_est_en;
extern int  init_map_size;
extern double imu_time_inte;
extern double laser_point_cov;
extern double acc_cov_input, gyr_cov_input, vel_cov;
extern double gyr_cov_output, acc_cov_output, b_gyr_cov, b_acc_cov;
extern double filter_size_surf_min, filter_size_map_min;
extern bool imu_en;
extern std::vector<double> gravity;
extern double G_m_s2;
extern double time_update_last, time_current, time_predict_last_const;

// --- Extrinsic (defined in voxelslam.cpp) ---
extern V3D Lidar_T_wrt_IMU;
extern M3D Lidar_R_wrt_IMU;
extern std::vector<double> extrinT;
extern std::vector<double> extrinR;

// --- IMU processor ---
extern std::shared_ptr<ImuProcess> p_imu;

// --- MeasureGroup ---
extern MeasureGroup Measures;

void readParameters(rclcpp::Node::SharedPtr node);
