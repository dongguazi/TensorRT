#pragma once
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <map>


// 中点坐标宽高
struct Bbox {
	float x;
	float y;
	float w;
	float h;
	float score;
	int classes;
};

//只需把box映射回原图
std::vector<Bbox> rescale_box(std::vector<Bbox>& out, int input_width, int input_height);
std::vector<cv::Rect> rescale_box(std::vector<cv::Rect>& out, int input_width, int input_height);
//可视化
cv::Mat renderBoundingBox(cv::Mat image, const std::vector<Bbox>& bboxes, std::vector<std::string> class_name);
