
#include <torch/torch.h> // One-stop header.
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <tuple>
#include <rclcpp/rclcpp.hpp>
#include <torchvision/vision.h>
#include "cslam_common_interfaces/msg/keyframe_rgb.hpp"
#include "cslam_common_interfaces/msg/global_descriptor.hpp"
#include <cv_bridge/cv_bridge.h>

#include <opencv2/cudaimgproc.hpp>

using std::placeholders::_1;
using namespace torch::jit;
using namespace cv::cuda;

const std::vector<float> IMAGENET_DEFAULT_MEAN {0.485, 0.456, 0.406};
const std::vector<float> IMAGENET_DEFAULT_STD {0.229, 0.224, 0.225};

namespace cslam {
    class Model : public script::Module {
        public:
            Model() {

            }

            Model(script::Module module) : script::Module(module) {

            }

            std::vector<char> get_the_bytes(std::string filename) {
                std::ifstream input(filename, std::ios::binary);
                std::vector<char> bytes(
                    (std::istreambuf_iterator<char>(input)),
                    (std::istreambuf_iterator<char>()));

                input.close();
                return bytes;
            }

            void load_parameters(std::string pt_pth) {
                std::vector<char> f = this->get_the_bytes(pt_pth);
                c10::Dict<c10::IValue, c10::IValue> weights = torch::pickle_load(f).toGenericDict();
                auto model_params = this->named_parameters();

                torch::NoGradGuard no_grad;
                for (auto const& w : weights) {
                    std::string name = w.key().toStringRef();
                    at::Tensor param = w.value().toTensor();
                    bool paramSet = false;
                    for (auto w : model_params) {
                        //std::cout << w.name << " was evaluated \n";
                        if (w.name == name) {
                            w.value.copy_(param);
                            paramSet = true;
                            break;
                            
                        }
                    }
                }
            }
    };


  class GlobalDescriptorComponent : public rclcpp::Node
  {
    public:

      GlobalDescriptorComponent(rclcpp::NodeOptions ops) : Node("global_descriptor_node", ops)
      {
          declare_parameter<int>("frontend.cosplace.descriptor_dim", 64);
          declare_parameter<std::string>("frontend.cosplace.backbone", "/models/cosplace_model_resnet18.pth");
          declare_parameter<std::string>("frontend.global_descriptor_technique", "cosplace");
          declare_parameter<std::string>("frontend.nn_checkpoint", "/models/resnet18_64.pth");
          
          try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            model = Model(torch::jit::load(get_parameter("frontend.cosplace.backbone").as_string()));
            model.load_parameters(get_parameter("frontend.nn_checkpoint").as_string());
            model.to(c10::kCUDA);
            model.eval();
          }
          catch (const c10::Error& e) {
            std::cerr << "error loading the model\n" << e.what();
          }

          global_descriptor_publisher = create_publisher<cslam_common_interfaces::msg::GlobalDescriptor>(
            "cslam/processed_global_descriptor", 100);

          keyframe_subscriber = create_subscription<
            cslam_common_interfaces::msg::KeyframeRGB>(
            "cslam/keyframe_data", 100,
            std::bind(&GlobalDescriptorComponent::receive_keyframe, this,
                        std::placeholders::_1));
      };

      private:
        rclcpp::Subscription<
            cslam_common_interfaces::msg::KeyframeRGB>::SharedPtr
            keyframe_subscriber;

        rclcpp::Publisher<
            cslam_common_interfaces::msg::GlobalDescriptor>::SharedPtr
            global_descriptor_publisher;

        bool initialized;
        GpuMat cvtColorMat, floatMat; 
        Stream stream;
        cslam::Model model;


        void receive_keyframe(const std::shared_ptr<const cslam_common_interfaces::msg::KeyframeRGB> keyframe_msg) {
            
            cv_bridge::CvImageConstPtr keyframe = cv_bridge::toCvShare(keyframe_msg->image, keyframe_msg);
            GpuMat image = GpuMat(keyframe->image);
            if (!initialized) {
                cvtColorMat = cv::cuda::GpuMat(image.size(), CV_8UC3);
                floatMat = cv::cuda::GpuMat(image.size(), CV_32FC3);
                initialized = true;
            }
            // convert RGB to grey-scale image in [0,1]
            if(image.channels() > 1) {
                cv::cuda::cvtColor(image, cvtColorMat, cv::COLOR_RGB2GRAY, 3, stream);
            } else {
                cv::cuda::cvtColor(image, cvtColorMat, cv::COLOR_GRAY2RGB, 3, stream);
            }

            cvtColorMat.convertTo(floatMat, CV_32FC3, 1.0 / 255.0, 0.0, stream);
           
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false);
            const at::Tensor tensorImage = torch::from_blob(floatMat.data, {floatMat.rows, floatMat.cols, 3}, options);
            auto permuted = tensorImage.permute({2, 0, 1});
            std::vector<torch::jit::IValue> input;
            input.push_back(permuted);
            at::Tensor output = model.forward(input).toTensor();
            auto embedding = output[0].detach().cpu().to(torch::kDouble).contiguous();
            auto globalDescriptorMsg = std::make_unique<cslam_common_interfaces::msg::GlobalDescriptor>();
            globalDescriptorMsg->descriptor = std::vector(embedding.const_data_ptr<double>(), embedding.const_data_ptr<double>() + embedding.numel());
            globalDescriptorMsg->keyframe_id = keyframe_msg->id;
            global_descriptor_publisher->publish(std::move(globalDescriptorMsg));
            

        }
  };
};

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(cslam::GlobalDescriptorComponent)