#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include "common_lib.h"
#include "parameters.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

// --- Process noise covariance ---
Eigen::Matrix<double, 30, 30> process_noise_cov_output();

// --- Process model f() for output model (30-DOF) ---
Eigen::Matrix<double, 30, 1> get_f_output(state_output &s, const input_ikfom &in);

// --- Jacobian df/dx for output model ---
Eigen::Matrix<double, 30, 30> df_dx_output(state_output &s, const input_ikfom &in);

// --- Transform point from body to world frame ---
void pointBodyToWorld(PointType const * const pi, PointType * const po);

// --- Publish odometry + TF at high frequency ---
void publish_odometry(
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pubOdomAftMapped,
    rclcpp::Node::SharedPtr node);

#endif
