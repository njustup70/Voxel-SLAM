#pragma once

#include "tools.hpp"
#include "ekf_imu.hpp"
#include "voxel_map.hpp"
#include "feature_point.hpp"
#include "loop_refine.hpp"
#include <mutex>
#include <Eigen/Eigenvalues>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <malloc.h>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include "BTC.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <livox_ros_driver2/msg/custom_msg.hpp>

using namespace std;

// Forward declarations for ROS 2 publishers
extern rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_scan, pub_cmap, pub_init, pub_pmap;
extern rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_test, pub_prev_path, pub_curr_path;

template <typename T>
void pub_pl_func(T &pl, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub, rclcpp::Node::SharedPtr node)
{
  pl.height = 1; pl.width = pl.size();
  sensor_msgs::msg::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "camera_init";
  output.header.stamp = node->now();
  pub->publish(output);
}

extern mutex mBuf;
extern Features feat;
extern deque<sensor_msgs::msg::Imu::SharedPtr> imu_buf;
extern deque<pcl::PointCloud<PointType>::Ptr> pcl_buf;
extern deque<double> time_buf;

extern double imu_last_time;
extern int point_notime;
extern double last_pcl_time;

void imu_handler(const sensor_msgs::msg::Imu::SharedPtr msg_in);
void pcl_handler_livox(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg);
void pcl_handler_standard(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

bool sync_packages(pcl::PointCloud<PointType>::Ptr &pl_ptr, deque<sensor_msgs::msg::Imu::SharedPtr> &imus, IMUEKF &p_imu);

double dept_err, beam_err;
void calcBodyVar(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &var);

// Compute the variance of the each point
void var_init(IMUST &ext, pcl::PointCloud<PointType> &pl_cur, PVecPtr pptr, double dept_err, double beam_err);

void pvec_update(PVecPtr pptr, IMUST &x_curr, PLV(3) &pwld);

// Read the alidarstate.txt
void read_lidarstate(string filename, vector<ScanPose*> &bl_tem);

double get_memory();

void icp_check(pcl::PointCloud<PointType> &pl_src, pcl::PointCloud<PointType> &pl_tar, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub_src, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub_tar, pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform, IMUST &xx, rclcpp::Node::SharedPtr node);
