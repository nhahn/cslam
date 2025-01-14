#include "cslam/front_end/rgbd_handler.h"
#include <rtabmap_conversions/MsgConversion.h>

using namespace rtabmap;
using namespace cslam;

template class cslam::RGBDHandler<ApproximateRGBDSync>;
template class cslam::RGBDHandler<ExactRGBDSync>;

template<typename SyncPolicy>
RGBDHandler<SyncPolicy>::RGBDHandler(rclcpp::Node * node) : SensorHandler(node) 
{
    node_->declare_parameter<std::string>("frontend.color_image_topic", "color/image");
    node_->declare_parameter<std::string>("frontend.depth_image_topic", "depth/image");
    node_->declare_parameter<std::string>("frontend.color_camera_info_topic",
                                          "color/camera_info");
    auto qos = rclcpp::SensorDataQoS().get_rmw_qos_profile();

    sub_image_color_.subscribe(
        node_, node_->get_parameter("frontend.color_image_topic").as_string(), "raw",
        qos);
    sub_image_depth_.subscribe(
        node_, node_->get_parameter("frontend.depth_image_topic").as_string(), "raw",
        qos);
    sub_camera_info_color_.subscribe(
        node_, node_->get_parameter("frontend.color_camera_info_topic").as_string(),
        qos);
    rgbd_synchronizer = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(max_queue_size_), sub_image_color_, sub_image_depth_,
        sub_camera_info_color_);
    if constexpr (std::is_same_v<SyncPolicy, ApproximateRGBDSync>) {
      rgbd_synchronizer->getPolicy()->setMaxIntervalDuration(rclcpp::Duration::from_seconds(node_->get_parameter("frontend.sync_period").as_double()));
    }

    rgbd_synchronizer->registerCallback(
        std::bind(&RGBDHandler::rgbd_callback, this, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3));
}

template<typename SyncPolicy>
void RGBDHandler<SyncPolicy>::rgbd_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr image_rect_rgb,
    const sensor_msgs::msg::Image::ConstSharedPtr image_rect_depth,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_rgb)
{

  rclcpp::Time stamp = rtabmap_conversions::timestampFromROS(image_rect_rgb->header.stamp) > rtabmap_conversions::timestampFromROS(image_rect_depth->header.stamp) ? image_rect_rgb->header.stamp : image_rect_depth->header.stamp;

  auto sensor_data = std::make_shared<rtabmap_msgs::msg::SensorData>();
  sensor_data->left = *image_rect_rgb;
  sensor_data->right = *image_rect_depth;
  sensor_data->header.stamp = stamp;
  sensor_data->left_camera_info.push_back(*camera_info_rgb);

  if(base_frame_id_.length() > 0) {
      geometry_msgs::msg::TransformStamped t;

      // Look up for the transformation between target_frame and turtle2 frames
      // and send velocity commands for turtle2 to reach target_frame
      try {
        t = tf_buffer_->lookupTransform(
          image_rect_rgb->header.frame_id, base_frame_id_, 
          image_rect_rgb->header.stamp);
          
        sensor_data->local_transform.push_back(t.transform);
      } catch (const tf2::TransformException & ex) {
        RCLCPP_INFO(
          node_->get_logger(), "Could not transform %s to %s: %s",
          image_rect_rgb->header.frame_id.c_str(), base_frame_id_.c_str(), ex.what());
        return;
      }
  } else {
    sensor_data->local_transform.push_back(geometry_msgs::msg::Transform());
  }

  sensor_queue_->add<0>(sensor_data);

}
