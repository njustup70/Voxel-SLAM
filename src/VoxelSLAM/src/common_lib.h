#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include "so3_math.h"
#include "esekfom.hpp"
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <deque>
#include <memory>
#include <sensor_msgs/msg/imu.hpp>

using namespace std;
using namespace Eigen;

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2;
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

MTK_BUILD_MANIFOLD(state_input,
((vect3, pos))
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, bg))
((vect3, ba))
((vect3, gravity))
);

MTK_BUILD_MANIFOLD(state_output,
((vect3, pos))
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, omg))
((vect3, acc))
((vect3, gravity))
((vect3, bg))
((vect3, ba))
);

MTK_BUILD_MANIFOLD(input_ikfom,
((vect3, acc))
((vect3, gyro))
);

#define PI_M (3.14159265358)
#define VEC_FROM_ARRAY(v)        v[0],v[1],v[2]
#define MAT_FROM_ARRAY(v)        v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType>    PointCloudXYZI;
typedef vector<PointType, Eigen::aligned_allocator<PointType>>  PointVector;
typedef Vector3d V3D;
typedef Matrix3d M3D;

struct MeasureGroup
{
  PointCloudXYZI::Ptr lidar;
  deque<sensor_msgs::msg::Imu::SharedPtr> imu;
  double lidar_beg_time;
  double lidar_end_time;
  MeasureGroup() : lidar(new PointCloudXYZI()), lidar_beg_time(0), lidar_end_time(0) {}
};

#endif
