
#include "cslam/front_end/utils/optical_flow.hpp"
 #include <vpi/OpenCVInterop.hpp>

#if defined(__x86_64__) || defined(_M_X64) || defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
    constexpr uint32_t VPI_BACKEND = VPI_BACKEND_CUDA;
    constexpr bool orin = false;
#else 
    constexpr uint32_t VPI_BACKEND = VPI_BACKEND_CUDA; //VPI_BACKEND_OFA | VPI_BACKEND_PVA | VPI_BACKEND_VIC;
    constexpr bool orin = true;
#endif

  
 #define CHECK_STATUS(STMT)                                      \
     do                                                          \
     {                                                           \
         VPIStatus status__ = (STMT);                            \
         if (status__ != VPI_SUCCESS)                            \
         {                                                       \
             char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];         \
             vpiGetLastStatusMessage(buffer, sizeof(buffer));    \
             std::ostringstream ss;                              \
             ss << vpiStatusGetName(status__) << ": " << buffer; \
             throw std::runtime_error(ss.str());                 \
         }                                                       \
     } while (0);
  
using namespace cslam;

OpticalFlow::OpticalFlow(int max, int level, int iterations, int window) : maxKeypoints(max), pyrLevel(level), iters(iterations), windowSize(window) {
    CHECK_STATUS(vpiStreamCreate(VPI_BACKEND_CPU | VPI_BACKEND_CUDA | VPI_BACKEND_PVA, &stream));
    CHECK_STATUS(vpiArrayCreate(maxKeypoints, VPI_ARRAY_TYPE_KEYPOINT_F32, VPI_BACKEND_CPU | VPI_BACKEND_CUDA | VPI_BACKEND_PVA, &prevFeatures));
    CHECK_STATUS(vpiArrayCreate(maxKeypoints, VPI_ARRAY_TYPE_KEYPOINT_F32, VPI_BACKEND_CPU | VPI_BACKEND_CUDA | VPI_BACKEND_PVA, &curFeatures));
    CHECK_STATUS(vpiArrayCreate(maxKeypoints, VPI_ARRAY_TYPE_U8, VPI_BACKEND_CPU | VPI_BACKEND_CUDA | VPI_BACKEND_PVA, &status));
        // Parameters we'll use. No need to change them on the fly, so just define them here.
    // We're using the default parameters.
    CHECK_STATUS(vpiInitOpticalFlowPyrLKParams(VPI_BACKEND, &lkParams));
    // lkParams.epsilon = 0.01;
    lkParams.windowDimension = window;
    lkParams.numIterations = iterations;
    lkParams.useInitialFlow = 0;
}

void OpticalFlow::initialize(const cv::Mat &cvFrame) {
    CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvFrame, cvFrame.channels() > 1? VPI_IMAGE_FORMAT_BGR8 : VPI_IMAGE_FORMAT_Y8_ER, VPI_BACKEND_CUDA, &imgTempFrame));
  
    // Create grayscale image representation of input.
    CHECK_STATUS(vpiImageCreate(cvFrame.cols, cvFrame.rows, VPI_IMAGE_FORMAT_U8, VPI_BACKEND_CUDA | VPI_BACKEND_PVA, &imgFrame));

    // Create the image pyramids used by the algorithm
    CHECK_STATUS(
        vpiPyramidCreate(cvFrame.cols, cvFrame.rows, VPI_IMAGE_FORMAT_U8, pyrLevel, 0.5, VPI_BACKEND_CUDA | VPI_BACKEND_PVA, &pyrPrevFrame));
    CHECK_STATUS(vpiPyramidCreate(cvFrame.cols, cvFrame.rows, VPI_IMAGE_FORMAT_U8, pyrLevel, 0.5, VPI_BACKEND_CUDA | VPI_BACKEND_PVA, &pyrCurFrame));
            // Create Optical Flow payload
    CHECK_STATUS(vpiCreateOpticalFlowPyrLK(VPI_BACKEND, cvFrame.cols, cvFrame.rows, VPI_IMAGE_FORMAT_U8, pyrLevel, 0.5,
                                    &optflow));
    CHECK_STATUS(vpiStreamSync(stream));
    std::cout << "Initialized! " << std::endl;
    initialized = true;
}

void updateTrackedKeypoints(const std::vector<cv::KeyPoint> &keypoints, VPIArray curKeypoints, int max) {
    VPIArrayData ptsData;
    CHECK_STATUS(vpiArrayLockData(curKeypoints, VPI_LOCK_READ_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &ptsData));
    VPIArrayBufferAOS &aosKeypoints = ptsData.buffer.aos;

    // if (keypoints.size() > max) {
    //     std::sort(keypoints.begin(), keypoints.end(), [](cv::KeyPoint a, cv::KeyPoint b) { return a.response > b.response; });
    // }
    
     // reorder the keypoints to keep the first 'max' with highest scores.
     std::vector<VPIKeypointF32> kpt;
     for (size_t i = 0; i < keypoints.size() && i < max; i++) {
        VPIKeypointF32 kp;
        kp.x = keypoints[i].pt.x; 
        kp.y = keypoints[i].pt.y;
        kpt.emplace_back(kp);
     }
     VPIKeypointF32 *kptData = reinterpret_cast<VPIKeypointF32 *>(aosKeypoints.data);
     std::copy(kpt.begin(), kpt.end(), kptData);
     // update keypoint array size.
     *aosKeypoints.sizePointer = kpt.size();
    // std::cout << "KPs:  " << kptData[1].x << " " << kptData[1].y << std::endl;
     vpiArrayUnlock(curKeypoints);
}

std::vector<std::pair<bool, cv::Point2f>> OpticalFlow::match (const cv::Mat &from, const cv::Mat &to, const std::vector<cv::KeyPoint> &keypoints) {
    updateBaseFrame(from, keypoints);
    return matchNextFrame(to);
}

bool OpticalFlow::updateBaseFrame(const cv::Mat &base, const std::vector<cv::KeyPoint> &keypoints) {
    const std::lock_guard<std::mutex> lock(inferenceLock);
    if (!initialized) {
        initialize(base);
    } else {
        //Reinit our arrays
        // vpiArrayDestroy(prevFeatures);
        // vpiArrayDestroy(curFeatures);
        // vpiArrayDestroy(status);
        CHECK_STATUS(vpiArrayCreate(maxKeypoints, VPI_ARRAY_TYPE_KEYPOINT_F32, VPI_BACKEND_CPU | VPI_BACKEND_CUDA | VPI_BACKEND_PVA, &prevFeatures));
        CHECK_STATUS(vpiArrayCreate(maxKeypoints, VPI_ARRAY_TYPE_KEYPOINT_F32, VPI_BACKEND_CPU | VPI_BACKEND_CUDA | VPI_BACKEND_PVA, &curFeatures));
        CHECK_STATUS(vpiArrayCreate(maxKeypoints, VPI_ARRAY_TYPE_U8, VPI_BACKEND_CPU | VPI_BACKEND_CUDA | VPI_BACKEND_PVA, &status));
    }

    updateTrackedKeypoints(keypoints, curFeatures, maxKeypoints);
    // Wrap frame into a VPIImage, reusing the existing imgFrame.
    CHECK_STATUS(vpiImageSetWrappedOpenCVMat(imgTempFrame, base));

    // Convert it to grayscale
    CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, imgTempFrame, imgFrame, NULL))

    // Generate a pyramid out of it
    CHECK_STATUS(vpiSubmitGaussianPyramidGenerator(stream, VPI_BACKEND, imgFrame, pyrCurFrame, VPI_BORDER_CLAMP));
    CHECK_STATUS(vpiStreamSync(stream));
    return true;
}

std::vector<std::pair<bool, cv::Point2f>> OpticalFlow::matchNextFrame (const cv::Mat &to) {
    const std::lock_guard<std::mutex> lock(inferenceLock);
    std::vector<std::pair<bool, cv::Point2f>> matches;

    std::swap(prevFeatures, curFeatures);
    std::swap(pyrPrevFrame, pyrCurFrame);

    // Wrap frame into a VPIImage, reusing the existing imgFrame.
    CHECK_STATUS(vpiImageSetWrappedOpenCVMat(imgTempFrame, to));

    // Convert it to grayscale
    CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, imgTempFrame, imgFrame, NULL))

    // Generate a pyramid out of it
    CHECK_STATUS(vpiSubmitGaussianPyramidGenerator(stream, VPI_BACKEND, imgFrame, pyrCurFrame, VPI_BORDER_CLAMP));

    // Estimate the features' position in current frame given their position in previous frame
    CHECK_STATUS(vpiSubmitOpticalFlowPyrLK(stream, 0, optflow, pyrPrevFrame, pyrCurFrame, prevFeatures,
                                        curFeatures, status, &lkParams));

    // Wait for processing to finish.
    CHECK_STATUS(vpiStreamSync(stream));

    // Lock the input and output arrays to draw the tracks to the output mask.
    VPIArrayData curFeaturesData, statusData;
    CHECK_STATUS(vpiArrayLockData(curFeatures, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &curFeaturesData));
    CHECK_STATUS(vpiArrayLockData(status, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &statusData));

    const VPIArrayBufferAOS &aosCurFeatures = curFeaturesData.buffer.aos;
    const VPIArrayBufferAOS &aosStatus      = statusData.buffer.aos;

    const VPIKeypointF32 *pCurFeatures = (VPIKeypointF32 *)aosCurFeatures.data;
    const uint8_t *pStatus             = (uint8_t *)aosStatus.data;

    int totKeypoints        = *curFeaturesData.buffer.aos.sizePointer;
    for (int i = 0; i < totKeypoints; i++)
    {
    matches.push_back(std::make_pair(pStatus[i] == 0, cv::Point2f(pCurFeatures[i].x, pCurFeatures[i].y)));
    }

    CHECK_STATUS(vpiArrayUnlock(curFeatures));
    CHECK_STATUS(vpiArrayUnlock(status));
  
    return matches;
}