#include <iostream>
#include <memory>
#include <tuple>
#include <filesystem>
#include <rclcpp/rclcpp.hpp>
#include "cslam_common_interfaces/msg/keyframe_rgb.hpp"
#include "cslam_common_interfaces/msg/global_descriptor.hpp"
#include <cv_bridge/cv_bridge.h>
#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include "lightglue_onnx/GeoNetOnnxRunner.hpp"


using std::placeholders::_1;
using namespace cv::cuda;

namespace cslam {

  class GlobalDescriptorComponent : public rclcpp::Node
  {
	public:

	  GlobalDescriptorComponent(rclcpp::NodeOptions ops) : Node("global_descriptor_node", ops)
	  {
		declare_parameter<std::string>("frontend.cosplace.model", "/models/cosplace_model_resnet18.pth");

		GlobalMatcher = std::make_shared<cslam::GeoNetOnnxRunner>(get_parameter("frontend.cosplace.model").as_string());

		global_descriptor_publisher = create_publisher<cslam_common_interfaces::msg::GlobalDescriptor>(
		"cslam/processed_global_descriptor", 100);

		keyframe_subscriber = create_subscription<
		cslam_common_interfaces::msg::KeyframeRGB>(
		"cslam/keyframe_data", 100,
		std::bind(&GlobalDescriptorComponent::receive_keyframe, this,
					std::placeholders::_1));
	  };

	  private:
    	std::shared_ptr<cslam::GeoNetOnnxRunner> GlobalMatcher;
		std::vector<float> embedding{};
		rclcpp::Subscription<
			cslam_common_interfaces::msg::KeyframeRGB>::SharedPtr
			keyframe_subscriber;

		rclcpp::Publisher<
			cslam_common_interfaces::msg::GlobalDescriptor>::SharedPtr
			global_descriptor_publisher;


		void receive_keyframe(const std::shared_ptr<const cslam_common_interfaces::msg::KeyframeRGB> keyframe_msg) {
			
			cv_bridge::CvImageConstPtr keyframe = cv_bridge::toCvShare(keyframe_msg->image, keyframe_msg);
			try{
        		GlobalMatcher->compute_embedding(keyframe->image, embedding);
				auto globalDescriptorMsg = std::make_unique<cslam_common_interfaces::msg::GlobalDescriptor>();
				for(int i = 0; i < embedding.size(); i++) {
					globalDescriptorMsg->descriptor.push_back(embedding[i]);
				}
				globalDescriptorMsg->keyframe_id = keyframe_msg->id; 
				global_descriptor_publisher->publish(std::move(globalDescriptorMsg));
			} catch (const std::exception &ex)
			{
				RCLCPP_WARN(get_logger(), "[ERROR] Global descriptor inference failed : %s", ex.what());
			}
		}
  };
};

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(cslam::GlobalDescriptorComponent)
