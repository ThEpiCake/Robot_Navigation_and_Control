#pragma once
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace arm_robot_control {
using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;
using Matrix4d = Eigen::Matrix4d;
using Vector6d = Eigen::Matrix<double, 6, 1>;

// Joint order: [theta1, d1, d2, theta4, theta5, theta6]
constexpr int THETA1 = 0, D1 = 1, D2 = 2, THETA4 = 3, THETA5 = 4, THETA6 = 5;

struct Pose { Matrix3d R{Matrix3d::Identity()}; Vector3d p{Vector3d::Zero()}; };
struct Frames { Pose base, joint1, joint2, joint3, wrist_center, end_effector; };
struct RobotConstants {
  double a1_z{0.02}, a2_z{0.15}, a3_z{0.38}, a4_z{0.11}, a5_z{0.30}, a6_z{0.04}, a7_z{0.05}, a8_z{0.11};
};
struct JointLimits { Vector6d lower, upper; };
struct IkOptions {
  double step_size{0.5}, tol_pos{1e-4}, tol_ori{1e-4}, damping_mu{0.03};
  int max_iterations{400};
  bool clamp_limits{true};
  double position_weight{1.0}, orientation_weight{1.0}, seed_regularization{0.0};
};

inline RobotConstants default_robot_constants() { return RobotConstants(); }
inline JointLimits default_joint_limits() {
  JointLimits l;
  l.lower << -M_PI, 0.0, 0.0, -M_PI, -1.5708, -M_PI;
  l.upper <<  M_PI, 0.255, 0.255, M_PI,  1.5708,  M_PI;
  return l;
}
inline Matrix3d fixed_xyz(double rx, double ry, double rz) {
  return (Eigen::AngleAxisd(rz, Vector3d::UnitZ()) * Eigen::AngleAxisd(ry, Vector3d::UnitY()) *
          Eigen::AngleAxisd(rx, Vector3d::UnitX())).toRotationMatrix();
}
inline Matrix3d exp_so3(const Vector3d & w) {
  const double th = w.norm();
  return (th < 1e-12) ? Matrix3d::Identity() : Eigen::AngleAxisd(th, w / th).toRotationMatrix();
}
inline Vector3d log_so3(const Matrix3d & R) {
  Eigen::AngleAxisd aa(R);
  if (std::abs(aa.angle()) < 1e-12) {
    return Vector3d::Zero();
  }
  return aa.axis() * aa.angle();
}
inline Vector3d orientation_error(const Matrix3d & R_current, const Matrix3d & R_target) {
  return log_so3(R_target * R_current.transpose());
}
inline Vector3d rpy_zyx_from_rotation(const Matrix3d & R) {
  const double yaw = std::atan2(R(1, 0), R(0, 0));
  const double pitch = std::atan2(-R(2, 0), std::sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2)));
  const double roll = std::atan2(R(2, 1), R(2, 2));
  return Vector3d(roll, pitch, yaw);
}
inline Vector6d clamp_to_joint_limits(const Vector6d & q, const JointLimits & lim) {
  Vector6d qc;
  for (int i = 0; i < 6; ++i) qc(i) = std::clamp(q(i), lim.lower(i), lim.upper(i));
  return qc;
}

inline Frames forward_kinematics(const Vector6d & joints, const RobotConstants & c = default_robot_constants()) {
  Frames f;
  const double a = std::sqrt(2.0) / 2.0;
  Matrix4d A1 = Matrix4d::Identity(), A2 = Matrix4d::Identity(), A3 = Matrix4d::Identity(), A4 = Matrix4d::Identity(),
           A5 = Matrix4d::Identity(), A6 = Matrix4d::Identity(), A7 = Matrix4d::Identity(), A8 = Matrix4d::Identity();
  A1 << std::cos(joints(THETA1)), -std::sin(joints(THETA1)), 0.0, 0.0,
        std::sin(joints(THETA1)),  std::cos(joints(THETA1)), 0.0, 0.0,
        0.0, 0.0, 1.0, c.a1_z,
        0.0, 0.0, 0.0, 1.0;
  A2(2, 3) = c.a2_z + joints(D1);
  A3 << -a, 0.0, a, 0.0,
         0.0, 1.0, 0.0, 0.0,
        -a, 0.0, -a, c.a3_z,
         0.0, 0.0, 0.0, 1.0;
  A4(2, 3) = c.a4_z + joints(D2);
  A5 << std::cos(joints(THETA4)), -std::sin(joints(THETA4)), 0.0, 0.0,
        std::sin(joints(THETA4)),  std::cos(joints(THETA4)), 0.0, 0.0,
        0.0, 0.0, 1.0, c.a5_z,
        0.0, 0.0, 0.0, 1.0;
  A6 << std::cos(joints(THETA5)), 0.0, std::sin(joints(THETA5)), 0.0,
        0.0, 1.0, 0.0, 0.0,
        -std::sin(joints(THETA5)), 0.0, std::cos(joints(THETA5)), c.a6_z,
        0.0, 0.0, 0.0, 1.0;
  A7 << std::cos(joints(THETA6)), -std::sin(joints(THETA6)), 0.0, 0.0,
        std::sin(joints(THETA6)),  std::cos(joints(THETA6)), 0.0, 0.0,
        0.0, 0.0, 1.0, c.a7_z,
        0.0, 0.0, 0.0, 1.0;
  A8(2, 3) = c.a8_z;
  const Matrix4d T01 = A1, T02 = T01 * A2, T03 = T02 * A3, T04 = T03 * A4;
  const Matrix4d T05 = T04 * A5, T06 = T05 * A6, T07 = T06 * A7, T08 = T07 * A8;
  f.base.R = T01.block<3, 3>(0, 0); f.base.p = T01.block<3, 1>(0, 3);
  f.joint1.R = T03.block<3, 3>(0, 0); f.joint1.p = T03.block<3, 1>(0, 3);
  f.joint2.R = T04.block<3, 3>(0, 0); f.joint2.p = T04.block<3, 1>(0, 3);
  f.joint3.R = T05.block<3, 3>(0, 0); f.joint3.p = T05.block<3, 1>(0, 3);
  f.wrist_center.R = T06.block<3, 3>(0, 0); f.wrist_center.p = T06.block<3, 1>(0, 3);
  f.end_effector.R = T08.block<3, 3>(0, 0); f.end_effector.p = T08.block<3, 1>(0, 3);
  return f;
}

inline bool solve_ik_analytic(
  const Pose & target_pose, Vector6d & joint_solution, const IkOptions & options,
  const RobotConstants & c = default_robot_constants(), const JointLimits & limits = default_joint_limits());

inline bool solve_ik_analytic(
  const Pose & target_pose, const Vector6d & seed, Vector6d & joint_solution, const IkOptions & options,
  const RobotConstants & c = default_robot_constants(), const JointLimits & limits = default_joint_limits()) {
  (void)seed;
  return solve_ik_analytic(target_pose, joint_solution, options, c, limits);
}

inline bool solve_ik_analytic(
  const Pose & target_pose, Vector6d & joint_solution, const IkOptions & options,
  const RobotConstants & c, const JointLimits & limits) {
  const double eps = 1e-9, rt2 = std::sqrt(2.0);
  // Step 1: position via Pc at O6 (consistent with 0A6):
  // Pc = p_ee - (a7+a8) * z_ee
  const Vector3d pc = target_pose.p - (c.a7_z + c.a8_z) * target_pose.R.col(2);
  const double xc = pc.x(), yc = pc.y(), zc = pc.z();
  const double K = std::sqrt(xc * xc + yc * yc);  // K>=0
  Vector6d joints_pos = Vector6d::Zero();
  joints_pos(THETA1) = (K < eps) ? 0.0 : std::atan2(yc, xc);
  // From p06:
  // x_c = (sqrt(2)/2) * cos(theta1) * (0.45 + d2)
  // y_c = (sqrt(2)/2) * sin(theta1) * (0.45 + d2)
  // z_c = 0.55 + d1 - (sqrt(2)/2) * (0.45 + d2)
  joints_pos(D2) = rt2 * K - 0.45;
  joints_pos(D1) = zc + (rt2 / 2.0) * joints_pos(D2) + (45.0 * rt2 / 200.0) - 11.0 / 20.0;
  // Step 2: orientation via R47 = (0R4)^T * R
  Vector6d joints_tmp = Vector6d::Zero();
  joints_tmp(THETA1) = joints_pos(THETA1); joints_tmp(D1) = joints_pos(D1); joints_tmp(D2) = joints_pos(D2);
  const Matrix3d R04 = forward_kinematics(joints_tmp, c).joint2.R;
  const Matrix3d R47 = R04.transpose() * target_pose.R;
  const double r13 = R47(0, 2), r23 = R47(1, 2), r31 = R47(2, 0), r32 = R47(2, 1), r33 = R47(2, 2);
  const double s5_abs = std::sqrt(std::max(0.0, r13 * r13 + r23 * r23));
  auto build_wrist_solution = [&](double s5) {
    Vector6d joints = joints_pos;
    // From R47 = Rz(theta4)*Ry(theta5)*Rz(theta6):
    // r13 = c4*s5, r23 = s4*s5, r33 = c5, r31 = -s5*c6, r32 = s5*s6
    joints(THETA5) = std::atan2(s5, r33);
    joints(THETA4) = std::atan2(r23 / s5, r13 / s5);
    joints(THETA6) = std::atan2(r32 / s5, -r31 / s5);
    return joints;
  };
  Vector6d selected_joint_values = joints_pos;
  if (s5_abs < eps) {
    // Project rule for singular case:
    selected_joint_values(THETA4) = 0.0;
    selected_joint_values(THETA5) = 0.0;
    selected_joint_values(THETA6) = 0.0;
  } else {
    // Project rule for branch selection: always choose positive s5.
    selected_joint_values = build_wrist_solution(+s5_abs);
  }
  if (options.clamp_limits) selected_joint_values = clamp_to_joint_limits(selected_joint_values, limits);
  joint_solution = selected_joint_values;
  const Pose fk = forward_kinematics(joint_solution, c).end_effector;
  const double ep = (target_pose.p - fk.p).norm(), eo = orientation_error(fk.R, target_pose.R).norm();
  return ep < options.tol_pos && eo < options.tol_ori;
}

inline bool solve_ik_iterative(
  const Pose & target, const Vector6d & seed, Vector6d & solution, const IkOptions & options,
  const RobotConstants & c = default_robot_constants(), const JointLimits & limits = default_joint_limits())
{
  const double eps = 1e-9, rt2 = std::sqrt(2.0);

  const Vector3d pc = target.p - (c.a7_z + c.a8_z) * target.R.col(2);
  const double xc = pc.x(), yc = pc.y(), zc = pc.z();
  const double K = std::sqrt(xc * xc + yc * yc);
  Vector6d joints_pos = Vector6d::Zero();
  joints_pos(THETA1) = (K < eps) ? 0.0 : std::atan2(yc, xc);
  joints_pos(D2) = rt2 * K - 0.45;
  joints_pos(D1) = zc + (rt2 / 2.0) * joints_pos(D2) + (45.0 * rt2 / 200.0) - 11.0 / 20.0;

  Vector6d joints_tmp = Vector6d::Zero();
  joints_tmp(THETA1) = joints_pos(THETA1);
  joints_tmp(D1)     = joints_pos(D1);
  joints_tmp(D2)     = joints_pos(D2);
  const Matrix3d R04 = forward_kinematics(joints_tmp, c).joint2.R;
  const Matrix3d R47 = R04.transpose() * target.R;
  const double r13 = R47(0, 2), r23 = R47(1, 2), r31 = R47(2, 0), r32 = R47(2, 1), r33 = R47(2, 2);
  const double s5_abs = std::sqrt(std::max(0.0, r13 * r13 + r23 * r23));

  auto build_wrist = [&](double s5) {
    Vector6d j = joints_pos;
    j(THETA5) = std::atan2(s5, r33);
    j(THETA4) = std::atan2(r23 / s5, r13 / s5);
    j(THETA6) = std::atan2(r32 / s5, -r31 / s5);
    return j;
  };

  Vector6d sol;
  if (s5_abs < eps) {
    // Singularity: preserve θ4+θ6 = seed[4]+seed[6], zero only θ5
    sol = joints_pos;
    sol(THETA4) = seed(THETA4);
    sol(THETA5) = 0.0;
    sol(THETA6) = seed(THETA6);
  } else {
    const Vector6d branch_pos = build_wrist(+s5_abs);
    const Vector6d branch_neg = build_wrist(-s5_abs);
    sol = ((branch_pos - seed).norm() <= (branch_neg - seed).norm()) ? branch_pos : branch_neg;
  }

  if (options.clamp_limits) sol = clamp_to_joint_limits(sol, limits);
  solution = sol;
  const Pose fk = forward_kinematics(solution, c).end_effector;
  const double ep = (target.p - fk.p).norm(), eo = orientation_error(fk.R, target.R).norm();
  return ep < options.tol_pos && eo < options.tol_ori;
}
}  // namespace arm_robot_control
