#pragma once
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <map>




// 转换图像数据: 先转换元素类型, (可选)然后归一化到[0, 1], (可选)然后交换RB通道
void convert(const cv::Mat& input, cv::Mat& output, const bool normalize, const bool exchangeRB);

//图像前处理,返回缩放比例
float preprocessWithRatio(const cv::Mat& input_image, float input_tensor[], int input_width, int input_height, int input_channels);

//前处理
void preprocess(cv::Mat& img, float* data, int model_width, int model_height, int model_channels);