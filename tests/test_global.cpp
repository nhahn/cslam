/**
 * @file main.cpp
 * @author letterso
 * @brief modified form OroChippw/LightGlue-OnnxRunner
 * @version 0.5
 * @date 2023-11-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/cudaarithm.hpp>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

namespace fs = std::filesystem;


#ifndef RT_CACHE_ROOT
	#define RT_CACHE_ROOT "/tmp"
#endif
#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)
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

  class GlobalDescriptorComponent
  {
	public:

	  GlobalDescriptorComponent(std::string model) {

		const auto& api = Ort::GetApi();

		env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LightGlueDecoupleOnnxRunner Extractor");

		session_options = Ort::SessionOptions();
		//session_options0.SetLogSeverityLevel(1);
		session_options.SetInterOpNumThreads(std::thread::hardware_concurrency());

		std::cout << "[INFO] OrtSessionOptions Append CUDAExecutionProvider" << std::endl;
		OrtCUDAProviderOptions cuda_options{};
		auto path = std::filesystem::path(model).parent_path() / "trt_engines";
		std::filesystem::create_directory(path);

		Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
					
		std::vector<const char*> option_keys = {
			"device_id",
			"trt_max_workspace_size",
			"trt_min_subgraph_size",
			"trt_fp16_enable",
			"trt_dla_enable",
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

		Session = std::make_unique<Ort::Session>(env, model.c_str(), session_options);

		// Initial Extractor
		size_t numInputNodes = Session->GetInputCount();
		InputNodeNames.reserve(numInputNodes);
		for (size_t i = 0; i < numInputNodes; i++)
		{
			InputNodeNames.emplace_back(strdup(Session->GetInputNameAllocated(i, allocator).get()));
			InputNodeShapes_.emplace_back(Session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
		}

        int srcInputTensorSize = InputNodeShapes_[0][0] * InputNodeShapes_[0][1] * InputNodeShapes_[0][2] * InputNodeShapes_[0][3];
        std::cout << "[INFO] Input size" << srcInputTensorSize << std::endl;

		size_t numOutputNodes = Session->GetOutputCount();
		OutputNodeNames.reserve(numOutputNodes);
		for (size_t i = 0; i < numOutputNodes; i++)
		{
			OutputNodeNames.emplace_back(strdup(Session->GetOutputNameAllocated(i, allocator).get()));
			OutputNodeShapes_.emplace_back(Session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
		}

		output.create(1, 64, CV_32F);
		gpu_allocator = Ort::Allocator(*Session.get(), pinned);
		binding = Ort::IoBinding(*Session.get());
		std::cout << "[INFO] ONNXRuntime environment created successfully." << std::endl;
		
	  };
      
      std::vector<float> receive_keyframe(const cv::Mat img) {
            std::cout << "uploading image";
			const int cropSize = MIN(img.cols, img.rows);
			const int offsetW = (img.cols - cropSize) / 2;
			const int offsetH = (img.rows - cropSize) / 2;
            std::cout << "crop" << offsetW << offsetH << cropSize << "\n";
			const cv::Rect centerCrop(offsetW, offsetH, cropSize, cropSize);
            img(centerCrop).copyTo(input);

			if (!initialized) {
                std::cout << "initializing";
				cvtColorMat = cv::cuda::GpuMat(centerCrop.size(), CV_8UC3);
				resizeMat = cv::cuda::GpuMat(COS_RESIZE, CV_8UC3);
				cuda_buffer_size = 3 * COS_RESIZE.area() * sizeof(float);
				if(CUDA_FAILED(cudaMalloc(&cuda_resource, cuda_buffer_size))) {
					throw new std::runtime_error("Cuda error!");
				}
				floatMat = cv::cuda::GpuMat(COS_RESIZE, CV_32FC3, cuda_resource);

				binding.BindInput("image",
					Ort::Value::CreateTensor<float>(
						memory_info_cuda, (float *) cuda_resource, 3 * COS_RESIZE.area() * sizeof(float),
						InputNodeShapes_[0].data(), InputNodeShapes_[0].size()));
				binding.BindOutput("embedding", 
                Ort::Value::CreateTensor<float>(
						memory_info_cuda, (float *) output.createGpuMatHeader().cudaPtr(), (size_t) output.size().area() * sizeof(float),
						OutputNodeShapes_[0].data(), OutputNodeShapes_[0].size()));
				std::cout << "Initialied global descriptor ONNX model";
				initialized = true;
			}

			// convert RGB to grey-scale image in [0,1]
			if(input.channels() > 1) {
				cv::cuda::cvtColor(input, cvtColorMat, cv::COLOR_RGB2GRAY, 3, stream);
			} else {
				cv::cuda::cvtColor(input, cvtColorMat, cv::COLOR_GRAY2RGB, 3, stream);
            }
			cv::cuda::resize(cvtColorMat, resizeMat, COS_RESIZE, 0, 0, 1, stream);
			resizeMat.convertTo(floatMat, CV_32FC3, 1.0 / 255.0, 0.0, stream);

			try {
				binding.SynchronizeInputs();
				Session->Run(Ort::RunOptions{nullptr}, binding);
				binding.SynchronizeOutputs();
                std::vector<float> retVal(64);
                output.createMatHeader().copyTo(retVal);
				return retVal;
			} catch (const std::exception &ex)
			{
				std::cerr << "[ERROR] Global descriptor inference failed : " << ex.what();
				return std::vector<float>();
			}


			// auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false);
			// const at::Tensor tensorImage = torch::from_blob(floatMat.data, {floatMat.rows, floatMat.cols, 3}, options);
			// auto permuted = tensorImage.permute({2, 0, 1});
			// std::vector<torch::jit::IValue> input;
			// input.push_back(permuted);
			// at::Tensor output = model.forward(input).toTensor();
			// auto embedding = output[0].detach().cpu().to(torch::kDouble).contiguous();
			// auto globalDescriptorMsg = std::make_unique<cslam_common_interfaces::msg::GlobalDescriptor>();
			// globalDescriptorMsg->descriptor = std::vector(embedding.const_data_ptr<double>(), embedding.const_data_ptr<double>() + embedding.numel());
			// globalDescriptorMsg->keyframe_id = keyframe_msg->id;
			// global_descriptor_publisher->publish(std::move(globalDescriptorMsg));
			

		}

	  private:

	    const Ort::MemoryInfo memory_info_cuda{"Cuda", OrtArenaAllocator, /*device_id*/0,
                                 OrtMemTypeDefault};
		const Ort::MemoryInfo pinned{"CudaPinned", OrtDeviceAllocator, 0, OrtMemTypeCPUOutput};
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

		bool initialized = false;
        HostMem input {HostMem::AllocType::SHARED}, output  {HostMem::AllocType::SHARED}; 
		GpuMat cvtColorMat, floatMat, resizeMat, inputR, inputG, inputB; 
		Stream stream;
		size_t cuda_buffer_size;				
		char *cuda_resource;
		Ort::IoBinding binding{nullptr};

		

		cudaError_t cudaCheckError(cudaError_t retval, const char* txt, const char* file, int line )
		{
		#if !defined(CUDA_TRACE)
			if( retval == cudaSuccess)
				return cudaSuccess;
		#endif

			if( retval != cudaSuccess )
			{
				printf(LOG_CUDA "   %s (error %u) (hex 0x%02X)\n", cudaGetErrorString(retval), retval, retval);
				printf(LOG_CUDA "   %s:%i\n", file, line);	
			}

			return retval;
		}
  };
};


inline bool fileExists(const std::string &filename)
{
    std::ifstream file(filename.c_str());
    return file.good();
}

std::vector<cv::Mat> ReadImage(std::vector<cv::String> image_filelist, bool grayscale = false)
{
    /*
    Func:
        Read an image from path as RGB or grayscale

    */
    int mode = cv::IMREAD_COLOR;
    if (grayscale)
    {
        mode = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
    }

    std::vector<cv::Mat> image_matlist;
    for (const auto &file : image_filelist)
    {
        std::cout << "[FILE INFO] : " << file << std::endl;
        cv::Mat image = cv::imread(file, mode);
        if (image.empty())
        {
            throw std::runtime_error("[ERROR] Could not read image at " + file);
        }
        if (!grayscale)
        {
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // BGR -> RGB
        }
        image_matlist.emplace_back(image);
    }

    return image_matlist;
}

int main(int argc, char *argv[])
{
    /* ****** CONFIG START ****** */


    fs::path image_path1 = fs::current_path() / "assets/DSC_0410.JPG";
    fs::path image_path2 = fs::current_path() / "assets/DSC_0411.JPG";
    fs::path save_path = fs::current_path() / "assets/";

    std::vector<cv::String> image_filelist1 {image_path1.string(), image_path1.string(), image_path1.string(), image_path2.string()};
    std::vector<cv::String> image_filelist2 {image_path2.string(), image_path2.string(), image_path2.string(), image_path1.string()};

    std::cout << "[INFO] => Building Image Matlist1" << std::endl;
    std::vector<cv::Mat> image_matlist1 = ReadImage(image_filelist1, true);
    /* ****** Load Cfg , Mode And Image End****** */

    /* ****** ONNX Infer Start****** */
    std::shared_ptr<cslam::GlobalDescriptorComponent> FeatureMatcher = std::make_shared<cslam::GlobalDescriptorComponent>("/models/cosplaceResNet18_64.onnx");

    auto iter1 = image_matlist1.begin();
    std::string mode = "COSPLACE";

    for (; iter1 != image_matlist1.end();
         ++iter1)
    {
        auto startTime = std::chrono::steady_clock::now();
        auto kpts = FeatureMatcher->receive_keyframe(*iter1);
        auto endTime = std::chrono::steady_clock::now();

        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << "[INFO] " << mode << " single picture whole process takes time : "
                  << elapsedTime << " ms" <<  "descriptor" << kpts[0] << " " << kpts[1] << " " << kpts[2] << " " << kpts[3] << std::endl;
    }
    /* ****** ONNX Infer End****** */


    // printf("[INFO] Decouple model extractor inference %zu images mean cost %.2f ms , matcher mean cost %.2f", image_filelist1.size(),
    //         (FeatureMatcher->GetTimer("extractor") / image_filelist1.size()), (FeatureMatcher->GetTimer("matcher") / image_filelist1.size()));
    

    return EXIT_SUCCESS;
}
