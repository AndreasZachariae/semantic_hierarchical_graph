/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2020 Shivang Patel
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Shivang Patel
 *
 * Reference tutorial:
 * https://navigation.ros.org/tutorials/docs/writing_new_nav2planner_plugin.html
 *********************************************************************/

#include <cmath>
#include <string>
#include <memory>
#include "nav2_util/node_utils.hpp"

#include "shg/shg_planner.hpp"

namespace shg
{

  void SHGPlanner::configure(
      rclcpp_lifecycle::LifecycleNode::SharedPtr parent,
      std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
      std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
  {
    node_ = parent;
    name_ = name;
    tf_ = tf;
    costmap_ = costmap_ros->getCostmap();
    global_frame_ = costmap_ros->getGlobalFrameID();

    client_ = node_->create_client<nav_msgs::srv::GetPlan>("shg/plan_path");
    while (!client_->wait_for_service(std::chrono::duration<int>(1)))
    {
      if (!rclcpp::ok())
      {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Interrupted while waiting for the service. Exiting.");
        return;
      }
      RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "service not available, waiting again...");
    }
  }

  void SHGPlanner::cleanup()
  {
    RCLCPP_INFO(
        node_->get_logger(), "CleaningUp plugin %s of type NavfnPlanner",
        name_.c_str());
  }

  void SHGPlanner::activate()
  {
    RCLCPP_INFO(
        node_->get_logger(), "Activating plugin %s of type NavfnPlanner",
        name_.c_str());
  }

  void SHGPlanner::deactivate()
  {
    RCLCPP_INFO(
        node_->get_logger(), "Deactivating plugin %s of type NavfnPlanner",
        name_.c_str());
  }

  void SHGPlanner::callPythonPlanner(const geometry_msgs::msg::PoseStamped &start,
                                     const geometry_msgs::msg::PoseStamped &goal)
  {
    auto node = rclcpp::Node::make_shared("minimal_client");
    RCLCPP_INFO(node->get_logger(), "Calling python planner");
    auto client = node->create_client<nav_msgs::srv::GetPlan>("shg/plan_path");
    while (!client->wait_for_service(std::chrono::seconds(1)))
    {
      if (!rclcpp::ok())
      {
        RCLCPP_ERROR(node->get_logger(), "client interrupted while waiting for service to appear.");
        return;
      }
      RCLCPP_INFO(node->get_logger(), "waiting for service to appear...");
    }
    auto request = std::make_shared<nav_msgs::srv::GetPlan::Request>();
    request->start = start;
    request->goal = goal;
    request->tolerance = 0.1;

    auto result_future = client->async_send_request(request);
    if (rclcpp::spin_until_future_complete(node, result_future) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_ERROR(node->get_logger(), "service call failed :(");
      return;
    }
    global_path_ = result_future.get()->plan;
  }

  nav_msgs::msg::Path SHGPlanner::createPlan(
      const geometry_msgs::msg::PoseStamped &start,
      const geometry_msgs::msg::PoseStamped &goal)
  {
    // Checking if the goal and start state is in the global frame
    if (start.header.frame_id != global_frame_)
    {
      RCLCPP_ERROR(
          node_->get_logger(), "Planner will only accept start position from %s frame",
          global_frame_.c_str());
      return global_path_;
    }

    if (goal.header.frame_id != global_frame_)
    {
      RCLCPP_INFO(
          node_->get_logger(), "Planner will only accept goal position from %s frame",
          global_frame_.c_str());
      return global_path_;
    }

    std::thread t(&SHGPlanner::callPythonPlanner, this, start, goal);

    // auto request = std::make_shared<nav_msgs::srv::GetPlan::Request>();
    // request->start = start;
    // request->goal = goal;
    // request->tolerance = 0.1;

    // auto result = client_->async_send_request(request, [this](rclcpp::Client<nav_msgs::srv::GetPlan>::SharedFuture future)
    //                                           { RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Received response");
    //                                             global_path_ = future.get()->plan; });

    t.join();

    if (global_path_.poses.size() == 0)
    {
      RCLCPP_ERROR(node_->get_logger(), "Path is empty");
      if (node_->now() - buffered_path_.header.stamp < rclcpp::Duration(5, 0))
      {
        RCLCPP_ERROR(node_->get_logger(), "Using buffered path");
        return buffered_path_;
      }
    }
    else
    {
      RCLCPP_INFO(node_->get_logger(), "Path recieved with " + std::to_string(global_path_.poses.size()) + " nodes: ");
      // for (auto &pose : global_path_.poses)
      // {
      //   RCLCPP_INFO(node_->get_logger(), "x: %f, y: %f", pose.pose.position.x, pose.pose.position.y);
      // }
    }

    buffered_path_ = global_path_;
    return global_path_;
  }

} // namespace shg

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(shg::SHGPlanner, nav2_core::GlobalPlanner)
