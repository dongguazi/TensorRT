#pragma once
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <map>


// �е�������
struct Bbox {
	float x;
	float y;
	float w;
	float h;
	float score;
	int classes;
};

//ֻ���boxӳ���ԭͼ
std::vector<Bbox> rescale_box(std::vector<Bbox>& out, int input_width, int input_height);
std::vector<cv::Rect> rescale_box(std::vector<cv::Rect>& out, int input_width, int input_height);
//���ӻ�
cv::Mat renderBoundingBox(cv::Mat image, const std::vector<Bbox>& bboxes, std::vector<std::string> class_name);
