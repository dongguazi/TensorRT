#pragma once
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <map>




// ת��ͼ������: ��ת��Ԫ������, (��ѡ)Ȼ���һ����[0, 1], (��ѡ)Ȼ�󽻻�RBͨ��
void convert(const cv::Mat& input, cv::Mat& output, const bool normalize, const bool exchangeRB);

//ͼ��ǰ����,�������ű���
float preprocessWithRatio(const cv::Mat& input_image, float input_tensor[], int input_width, int input_height, int input_channels);

//ǰ����
void preprocess(cv::Mat& img, float* data, int model_width, int model_height, int model_channels);