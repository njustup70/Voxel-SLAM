#include "Estimator.h"
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

Eigen::Matrix<double, 30, 30> process_noise_cov_output()
{
  Eigen::Matrix<double, 30, 30> cov;
  cov.setZero();
  cov.block<3, 3>(12, 12).diagonal() << vel_cov, vel_cov, vel_cov;
  cov.block<3, 3>(15, 15).diagonal() << gyr_cov_output, gyr_cov_output, gyr_cov_output;
  cov.block<3, 3>(18, 18).diagonal() << acc_cov_output, acc_cov_output, acc_cov_output;
  cov.block<3, 3>(24, 24).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
  cov.block<3, 3>(27, 27).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
  return cov;
}

Eigen::Matrix<double, 30, 1> get_f_output(state_output &s, const input_ikfom &in)
{
  Eigen::Matrix<double, 30, 1> res = Eigen::Matrix<double, 30, 1>::Zero();
  vect3 a_inertial = s.rot * s.acc;
  for(int i = 0; i < 3; i++){
    res(i) = s.vel[i];
    res(i + 3) = s.omg[i];
    res(i + 12) = a_inertial[i] + s.gravity[i];
  }
  return res;
}

Eigen::Matrix<double, 30, 30> df_dx_output(state_output &s, const input_ikfom &in)
{
  Eigen::Matrix<double, 30, 30> cov = Eigen::Matrix<double, 30, 30>::Zero();
  cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
  cov.template block<3, 3>(12, 3) = -s.rot * MTK::hat(s.acc);
  cov.template block<3, 3>(12, 18) = -s.rot;
  cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity();
  cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();
  return cov;
}

void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_world;
  if (\!use_imu_as_input)
  {
    p_world = kf_output.x_.rot * (kf_output.x_.offset_R_L_I * p_body + kf_output.x_.offset_T_L_I) + kf_output.x_.pos;
  }
  else
  {
    p_world = kf_input.x_.rot * (kf_input.x_.offset_R_L_I * p_body + kf_input.x_.offset_T_L_I) + kf_input.x_.pos;
  }
  po->x = p_world(0);
  po->y = p_world(1);
  po->z = p_world(2);
  po->intensity = pi->intensity;
}

void publish_odometry(
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pubOdomAftMapped,
    rclcpp::Node::SharedPtr node)
{
  nav_msgs::msg::Odometry odom;
  odom.header.frame_id = "camera_init";
  odom.child_frame_id = "aft_mapped";
  odom.header.stamp = node->now();

  if (use_imu_as_input)
  {
    odom.pose.pose.position.x = kf_input.x_.pos(0);
    odom.pose.pose.position.y = kf_input.x_.pos(1);
    odom.pose.pose.position.z = kf_input.x_.pos(2);
    Eigen::Quaterniond q(kf_input.x_.rot);
    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.twist.twist.linear.x = kf_input.x_.vel(0);
    odom.twist.twist.linear.y = kf_input.x_.vel(1);
    odom.twist.twist.linear.z = kf_input.x_.vel(2);
  }
  else
  {
    odom.pose.pose.position.x = kf_output.x_.pos(0);
    odom.pose.pose.position.y = kf_output.x_.pos(1);
    odom.pose.pose.position.z = kf_output.x_.pos(2);
    Eigen::Quaterniond q(kf_output.x_.rot);
    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.twist.twist.linear.x = kf_output.x_.vel(0);
    odom.twist.twist.linear.y = kf_output.x_.vel(1);
    odom.twist.twist.linear.z = kf_output.x_.vel(2);
  }

  pubOdomAftMapped->publish(odom);

  // Also publish TF
  static std::shared_ptr<tf2_ros::TransformBroadcaster> br =
      std::make_shared<tf2_ros::TransformBroadcaster>(node);
  geometry_msgs::msg::TransformStamped ts;
  ts.header.stamp = odom.header.stamp;
  ts.header.frame_id = "camera_init";
  ts.child_frame_id = "aft_mapped";
  ts.transform.translation.x = odom.pose.pose.position.x;
  ts.transform.translation.y = odom.pose.pose.position.y;
  ts.transform.translation.z = odom.pose.pose.position.z;
  ts.transform.rotation = odom.pose.pose.orientation;
  br->sendTransform(ts);
}
