#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Omni_InverseKinematics.h"

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/float64.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

using namespace std::chrono_literals;

class KinematicsPublisher : public rclcpp::Node
{
public:
  KinematicsPublisher()
      : Node("abu_kinematics"), Vx_(0.0), Vy_(0.0), W_(0.0)
  {
    // Declare and get the topic name parameter
    input_topic_ = this->declare_parameter<std::string>("input_topic_kinematics", "cmd_vel");
    imu_topic_ = this->declare_parameter<std::string>("input_topic_imu", "imu");
    output_topic_ = this->declare_parameter<std::string>("output_topic_kinematics", "wheel_speed");
    button_topic_ = this->declare_parameter<std::string>("input_topic_button", "joy");
    
    // Create publisher and subscriber using the parameterized topic names
    pub_vel = this->create_publisher<std_msgs::msg::Float64MultiArray>(output_topic_, 10);
    sub_cmd_vel = this->create_subscription<geometry_msgs::msg::Twist>(
        input_topic_, 10, std::bind(&KinematicsPublisher::cmdvel_callback, this, std::placeholders::_1));

    sub_imu = this->create_subscription<std_msgs::msg::Float64>(
        imu_topic_, 10, std::bind(&KinematicsPublisher::imu_callback, this, std::placeholders::_1));

    sub_button = this->create_subscription<sensor_msgs::msg::Joy>(
        button_topic_, 10, std::bind(&KinematicsPublisher::button_topic_, this, std::placeholders::_1));

    timer_ = this->create_wall_timer(10ms, std::bind(&KinematicsPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {

    // Arrow control
    if (joy_arrow_[0] == 1)
    {
      target_heading_ = 270.0; // Rotate right
    }
    else if (joy_arrow_[0] == -1)
    {
      target_heading_ = 90.0; // Rotate left
    }
    else if (joy_arrow_[1] == 1)
    {
      target_heading_ = 0.0; // Move forward
    }
    else if (joy_arrow_[1] == -1)
    {
      target_heading_ = 180.0; // Move backward
    }
    else
    {
      target_heading_ = robot_heading_; // Maintain current heading
    }

    // Rotation PID
    
    double error = target_heading_ - robot_heading_;

    if (error > 180.0)
    {
      error -= 360.0;
    }
    else if (error < -180.0)
    {
      error += 360.0;
    }

    // PID control
    W_ = (error * Kp) + (integral_ * Ki) + ((error - previous_error_) * Kd);
    previous_error_ = error;
    integral_ += error;

    //integral windup prevention
    integral_ = std::clamp(integral_, -10.0, 10.0); // Limit integral windup

    // solve inverse kinematics
    wheel_speed_ = IK.solve(Vx_, Vy_, W_);

    // publish wheel speed
    auto vel_msg = std_msgs::msg::Float64MultiArray();
    vel_msg.data = {map_(wheel_speed_[0], -27.0, 27.0, -255, 255), map_(wheel_speed_[1], -27.0, 27.0, -255, 255), map_(wheel_speed_[2], -27.0, 27.0, -255, 255)};
    pub_vel->publish(vel_msg);

    // get parameter
    input_topic_ = this->get_parameter("input_topic_kinematics").as_string();
    imu_topic_ = this->get_parameter("input_topic_imu").as_string();
    button_topic_ = this->get_parameter("input_topic_button").as_string();
    output_topic_ = this->get_parameter("output_topic_kinematics").as_string();
  }

  void cmdvel_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    Vx_ = msg->linear.x;
    Vy_ = msg->linear.y;
    // W_ = msg->angular.z;
  }

  void imu_callback(const std_msgs::msg::Float64::SharedPtr msg)
  {
    // IMU data
    robot_heading_ = msg->data;
  }

  void button_topic_(const sensor_msgs::msg::Joy::SharedPtr msg)
  {
    // Button data
    joy_arrow_[0] = msg->axes[6];
    joy_arrow_[1] = msg->axes[7];
  }

  double map_(double x, double in_min, double in_max, double out_min, double out_max)
  {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
  }

  InverseKinematic IK{0.127f, 0.3f}; // wheel radius, robot radius (cm)
  double Vx_, Vy_, W_;
  std::vector<double> wheel_speed_ = {0.0, 0.0, 0.0};
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_vel;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_vel;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr sub_imu;

  // Arrow joy
  std::vector<int8_t> joy_arrow_ = {0.0, 0.0};

  // Parameter
  std::string input_topic_;
  std::string output_topic_;
  std::string imu_topic_;
  std::string button_topic_;
  double target_heading_ = 0.0; //deg 
  double robot_heading_ = 0.0;  //deg

  // PID parameters for rotation
  double Kp = 0.5; 
  double Ki = 0.0;  
  double Kd = 0.0; 

  //PID variables
  double previous_error_ = 0.0;
  double error_ = 0.0;
  double integral_ = 0.0;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KinematicsPublisher>());
  rclcpp::shutdown();
  return 0;
}