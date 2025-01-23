#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/OpticalFlowPyrLK.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstring> // for memset
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <vector>

namespace cslam {
    class OpticalFlow {
        public:
            explicit OpticalFlow(int max = 1024, int level = 3, int iterations = 30, int window = 15, bool usePVA = false);
            ~OpticalFlow() {
                vpiStreamDestroy(stream);
                vpiPayloadDestroy(optflow);
                vpiArrayDestroy(prevFeatures);
                vpiArrayDestroy(curFeatures);
                vpiArrayDestroy(status);
                if (initialized) {
                    vpiPyramidDestroy(pyrPrevFrame);
                    vpiImageDestroy(imgTempFrame);
                    vpiImageDestroy(imgFrame);
                }


            }
            std::vector<std::pair<bool, cv::Point2f>> match (const cv::Mat &from, const cv::Mat &to, const std::vector<cv::KeyPoint> &keypoints);
            bool updateBaseFrame(const cv::Mat &from, const std::vector<cv::KeyPoint> &keypoints);
            std::vector<std::pair<bool, cv::Point2f>> matchNextFrame (const cv::Mat &to);

        private:
            void initialize(const cv::Mat &cvFrame);
            int maxKeypoints = 1024, pyrLevel = 3, iters = 30, windowSize = 11;
            VPIOpticalFlowPyrLKParams lkParams;
            VPIStream stream        = NULL;
            VPIImage imgTempFrame   = NULL;
            VPIImage imgFrame       = NULL;
            VPIPyramid pyrPrevFrame = NULL, pyrCurFrame = NULL;
            VPIArray prevFeatures = NULL, curFeatures = NULL, status = NULL;
            VPIPayload optflow = NULL;
            uint32_t VPI_BACKEND = VPI_BACKEND_CUDA;
            bool initialized = false;
            std::mutex inferenceLock;

    };
}