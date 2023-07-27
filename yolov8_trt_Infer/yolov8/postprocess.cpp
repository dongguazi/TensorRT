#include "postprocess.h"


//只需把box映射回原图
std::vector<Bbox> rescale_box(std::vector<Bbox>& out, int input_width, int input_height) {
	float gain = 640.0 / std::max(input_width, input_height);
	float pad_x = (640.0 - input_width * gain) / 2;
	float pad_y = (640.0 - input_height * gain) / 2;

	std::vector<Bbox> boxs;
	Bbox box;
	for (int i = 0; i < (int)out.size(); i++) {
		box.x = (out[i].x - pad_x) / gain;
		box.y = (out[i].y - pad_y) / gain;
		box.w = out[i].w / gain;
		box.h = out[i].h / gain;
		box.score = out[i].score;
		box.classes = out[i].classes;

		boxs.push_back(box);
	}
	return boxs;
}

//只需把box映射回原图
std::vector<cv::Rect> rescale_box(std::vector<cv::Rect>& out, int input_width, int input_height) {
	float gain = 640.0 / std::max(input_width, input_height);
	float pad_x = (640.0 - input_width * gain) / 2;
	float pad_y = (640.0 - input_height * gain) / 2;

	std::vector<cv::Rect> boxs;
	cv::Rect box;

	for (int i = 0; i < (int)out.size(); i++) {
		box.x = (out[i].x - pad_x) / gain;
		box.y = (out[i].y - pad_y) / gain;
		box.width = out[i].width / gain;
		box.height = out[i].height / gain;

		boxs.push_back(box);
	}
	return boxs;
}

//可视化
cv::Mat renderBoundingBox(cv::Mat image, const std::vector<Bbox>& bboxes, std::vector<std::string> class_names) {
	for (const auto& rect : bboxes)
	{

		cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
		cv::rectangle(image, rst, cv::Scalar(255, 204, 0), 2, cv::LINE_8, 0);
		//cv::rectangle(image, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2), cv::Point(rect.x + rect.w / 2, rect.y + rect.h / 2), cv::Scalar(255, 204,0), 3);

		int baseLine;
		std::string label = class_names[rect.classes] + ": " + std::to_string(rect.score * 100).substr(0, 4) + "%";

		cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_TRIPLEX, 0.5, 1, &baseLine);
		//int newY = std::max(rect.y, labelSize.height);
		rectangle(image, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - round(1.5 * labelSize.height)),
			cv::Point(rect.x - rect.w / 2 + round(1.0 * labelSize.width), rect.y - rect.h / 2 + baseLine), cv::Scalar(255, 204, 0), cv::FILLED);
		cv::putText(image, label, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 204, 255));


	}
	return image;
}
