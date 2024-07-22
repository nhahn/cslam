#ifndef RT_CACHE_ROOT
	#define RT_CACHE_ROOT "/tmp"
#endif
#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)

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

#define CUDA(x)				cudaCheckError((x), #x, __FILE__, __LINE__)

/**
 * Evaluates to true on success
 * @ingroup cudaError
 */
#define CUDA_SUCCESS(x)			(CUDA(x) == cudaSuccess)

/**
 * Evaluates to true on failure
 * @ingroup cudaError
 */
#define CUDA_FAILED(x)			(CUDA(x) != cudaSuccess)

/**
 * Return from the boolean function if CUDA call fails
 * @ingroup cudaError
 */
#define CUDA_VERIFY(x)			if(CUDA_FAILED(x))	return false;

/**
 * LOG_CUDA string.
 * @ingroup cudaError
 */
#define LOG_CUDA "[cuda]   "



using std::placeholders::_1;
using namespace cv::cuda;

static const cv::Size COS_RESIZE {224, 224};
namespace cslam {

  class GlobalDescriptorComponent : public rclcpp::Node
  {
	public:

	  GlobalDescriptorComponent(rclcpp::NodeOptions ops) : Node("global_descriptor_node", ops)
	  {
		declare_parameter<int>("frontend.cosplace.descriptor_dim", 64);
		declare_parameter<std::string>("frontend.cosplace.model", "/models/cosplace_model_resnet18.pth");
		declare_parameter<std::string>("frontend.global_descriptor_technique", "cosplace");
			
		const auto& api = Ort::GetApi();

		env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Global Descriptor");

		session_options = Ort::SessionOptions();
		//session_options0.SetLogSeverityLevel(1);
		session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
		session_options.SetInterOpNumThreads(std::thread::hardware_concurrency());
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		auto modelPath = get_parameter("frontend.cosplace.model").as_string();
		std::cout << "[INFO] OrtSessionOptions Append CUDAExecutionProvider" << std::endl;
		OrtCUDAProviderOptions cuda_options{};
		auto path = std::filesystem::path(modelPath).parent_path() / "trt_engines";
		std::filesystem::create_directory(path);

		Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
					
		std::vector<const char*> option_keys = {
			"device_id",
			"trt_max_workspace_size",
			"trt_min_subgraph_size",
			"trt_fp16_enable",
			"trt_dla_enable",
			"trt_dla_core",
			// below options are strongly recommended !
			"trt_engine_cache_enable",
			"trt_engine_cache_path",
			"trt_timing_cache_enable",
			"trt_timing_cache_path",
		};
		std::vector<const char*> option_values = {
			"0",
			"2147483648",
			"5",
			"1",
			"1",
			"1",
			"1",
			path.c_str(),
			"1",
			path.c_str(), // can be same as the engine cache folder
		};

		Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(tensorrt_options,
															option_keys.data(), option_values.data(), option_keys.size()));


		// this implicitly sets "has_user_compute_stream"
		Ort::ThrowOnError(api.UpdateTensorRTProviderOptionsWithValue(tensorrt_options, "user_compute_stream", stream.cudaPtr()));

		/// below code can be used to print all options

		cuda_options.device_id = 0;
		cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
		cuda_options.gpu_mem_limit = 0;
		cuda_options.arena_extend_strategy = 1;     
		cuda_options.do_copy_in_default_stream = 1; 
		cuda_options.has_user_compute_stream = 1;
		cuda_options.user_compute_stream = stream.cudaPtr();
		cuda_options.default_memory_arena_cfg = nullptr;
		
		session_options.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);
		session_options.AppendExecutionProvider_CUDA(cuda_options);
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

		Session = std::make_unique<Ort::Session>(env, modelPath.c_str(), session_options);

		// Initial Extractor
		size_t numInputNodes = Session->GetInputCount();
		InputNodeNames.reserve(numInputNodes);
		for (size_t i = 0; i < numInputNodes; i++)
		{
			InputNodeNames.emplace_back(strdup(Session->GetInputNameAllocated(i, allocator).get()));
			InputNodeShapes_.emplace_back(Session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
		}

		size_t numOutputNodes = Session->GetOutputCount();
		OutputNodeNames.reserve(numOutputNodes);
		for (size_t i = 0; i < numOutputNodes; i++)
		{
			OutputNodeNames.emplace_back(strdup(Session->GetOutputNameAllocated(i, allocator).get()));
			OutputNodeShapes_.emplace_back(Session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
		}
		output.create(1, get_parameter("frontend.cosplace.descriptor_dim").as_int(), CV_32F);
		gpu_allocator = Ort::Allocator(*Session.get(), pinned);
		binding = Ort::IoBinding(*Session.get());
		std::cout << "[INFO] ONNXRuntime environment created successfully." << std::endl;
		
		global_descriptor_publisher = create_publisher<cslam_common_interfaces::msg::GlobalDescriptor>(
		"cslam/processed_global_descriptor", 100);

		keyframe_subscriber = create_subscription<
		cslam_common_interfaces::msg::KeyframeRGB>(
		"cslam/keyframe_data", 100,
		std::bind(&GlobalDescriptorComponent::receive_keyframe, this,
					std::placeholders::_1));
	  };

	  private:

	    const Ort::MemoryInfo memory_info_cuda{"Cuda", OrtArenaAllocator, /*device_id*/0,
                                 OrtMemTypeDefault};
		const Ort::MemoryInfo pinned{"CudaPinned", OrtArenaAllocator, 0, OrtMemTypeCPUOutput};
		Ort::Allocator gpu_allocator{nullptr};
		Ort::Env env;
		Ort::SessionOptions session_options;
		std::unique_ptr<Ort::Session> Session;
		Ort::AllocatorWithDefaultOptions allocator;
		OrtTensorRTProviderOptionsV2* tensorrt_options;

		std::vector<char *> InputNodeNames;
		std::vector<std::vector<int64_t>> InputNodeShapes_;
		std::vector<char *> OutputNodeNames;
		std::vector<std::vector<int64_t>> OutputNodeShapes_;

		rclcpp::Subscription<
			cslam_common_interfaces::msg::KeyframeRGB>::SharedPtr
			keyframe_subscriber;

		rclcpp::Publisher<
			cslam_common_interfaces::msg::GlobalDescriptor>::SharedPtr
			global_descriptor_publisher;

		bool initialized = false;
        HostMem input {HostMem::AllocType::SHARED}, output  {HostMem::AllocType::SHARED}; 
		GpuMat cvtColorMat, floatMat, resizeMat, inputR, inputG, inputB; 
		Stream stream;
		size_t cuda_buffer_size;				
		char *cuda_resource;
		Ort::IoBinding binding{nullptr};

		void receive_keyframe(const std::shared_ptr<const cslam_common_interfaces::msg::KeyframeRGB> keyframe_msg) {
			
			cv_bridge::CvImageConstPtr keyframe = cv_bridge::toCvShare(keyframe_msg->image, keyframe_msg);
			const cv::Mat img = keyframe->image;
			const int cropSize = MIN(img.cols, img.rows);
			const int offsetW = (img.cols - cropSize) / 2;
			const int offsetH = (img.rows - cropSize) / 2;
			const cv::Rect centerCrop(offsetW, offsetH, cropSize, cropSize);
            img(centerCrop).copyTo(input);

			if (!initialized) {
				cvtColorMat = cv::cuda::GpuMat(centerCrop.size(), CV_8UC3);
				resizeMat = cv::cuda::GpuMat(COS_RESIZE, CV_8UC3);
				cuda_buffer_size = 3 * COS_RESIZE.area() * sizeof(float);
				if(CUDA_FAILED(cudaMalloc(&cuda_resource, cuda_buffer_size))) {
					throw new std::runtime_error("Cuda error!");
				}
				floatMat = cv::cuda::GpuMat(COS_RESIZE, CV_32FC1);
				inputR = cv::cuda::GpuMat(COS_RESIZE, CV_32FC1, cuda_resource);
				inputG = cv::cuda::GpuMat(COS_RESIZE, CV_32FC1, cuda_resource + COS_RESIZE.area() * sizeof(float));
				inputB = cv::cuda::GpuMat(COS_RESIZE, CV_32FC1, cuda_resource + COS_RESIZE.area() + sizeof(float) * 2);

				binding.BindInput("image",
					Ort::Value::CreateTensor<float>(
						memory_info_cuda, (float *) cuda_resource, 3 * COS_RESIZE.area() * sizeof(float),
						InputNodeShapes_[0].data(), InputNodeShapes_[0].size()));
				binding.BindOutput("embedding", 
                Ort::Value::CreateTensor<float>(
						memory_info_cuda, (float *) output.createGpuMatHeader().cudaPtr(), (size_t) output.size().area() * sizeof(float),
						OutputNodeShapes_[0].data(), OutputNodeShapes_[0].size()));
				RCLCPP_INFO(get_logger(), "Initialied global descriptor ONNX model");
				initialized = true;
			}

			// convert RGB to grey-scale image in [0,1]
			if(input.channels() > 1) {
				cv::cuda::cvtColor(input, cvtColorMat, cv::COLOR_RGB2GRAY, 1, stream);
				cv::cuda::resize(cvtColorMat, resizeMat, COS_RESIZE, 0, 0, 1, stream);
				resizeMat.convertTo(floatMat, CV_32FC1, 1.0 / 255.0, 0.0, stream);
			} else {
				cv::cuda::resize(input, resizeMat, COS_RESIZE, 0, 0, 1, stream);
				resizeMat.convertTo(floatMat, CV_32FC1, 1.0 / 255.0, 0.0, stream);
            }

			cv::cuda::subtract(floatMat, 0.485, inputR, cv::noArray(), -1, stream); cv::cuda::divide(inputR, 0.229, inputR, 1.0, -1, stream);
			cv::cuda::subtract(floatMat, 0.456, inputG, cv::noArray(), -1, stream); cv::cuda::divide(inputG, 0.224, inputG, 1.0, -1, stream);
			cv::cuda::subtract(floatMat, 0.406, inputB, cv::noArray(), -1, stream); cv::cuda::divide(inputB, 0.225, inputB, 1.0, -1, stream);

			try {
				binding.SynchronizeInputs();
				Session->Run(Ort::RunOptions{nullptr}, binding);
				binding.SynchronizeOutputs();
				auto globalDescriptorMsg = std::make_unique<cslam_common_interfaces::msg::GlobalDescriptor>();
				output.createMatHeader().convertTo(globalDescriptorMsg->descriptor, CV_64F);
				globalDescriptorMsg->keyframe_id = keyframe_msg->id; 
				global_descriptor_publisher->publish(std::move(globalDescriptorMsg));
			} catch (const std::exception &ex)
			{
				RCLCPP_WARN(get_logger(), "[ERROR] Global descriptor inference failed : %s", ex.what());
			}

		}

		cudaError_t cudaCheckError(cudaError_t retval, const char* txt, const char* file, int line )
		{
		#if !defined(CUDA_TRACE)
			if( retval == cudaSuccess)
				return cudaSuccess;
		#endif

			if( retval != cudaSuccess )
			{
				RCLCPP_ERROR(get_logger(), LOG_CUDA "   %s (error %u) (hex 0x%02X)\n", cudaGetErrorString(retval), retval, retval);
				RCLCPP_ERROR(get_logger(), LOG_CUDA "   %s:%i\n", file, line);	
			}

			return retval;
		}
  };
};

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(cslam::GlobalDescriptorComponent)
