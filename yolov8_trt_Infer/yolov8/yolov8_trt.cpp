// xujing
//YOLOv8


#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdio.h>
#include <math.h>
#include <string>

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

#include <npp.h>
#include "logging.h"

using namespace sample;
using namespace std;
using namespace cv;

#define BATCH_SIZE 1
#define INPUT_W 640
#define INPUT_H 640
#define INPUT_SIZE 640

#define NUMS_CLASS 7
#define OUTPUT_SIZE 8400

#define IsPadding 1


std::vector<std::string> class_names = { "rubber stopper", "push rod tail", "needle tail", "mouth", "crooked mouth", "screw mouth", "small rubber plug" };


// �е�������
struct Bbox {
	float x;
	float y;
	float w;
	float h;
	float score;
	int classes;
};

/// ת��ͼ������: ��ת��Ԫ������, (��ѡ)Ȼ���һ����[0, 1], (��ѡ)Ȼ�󽻻�RBͨ��
void convert(const cv::Mat& input, cv::Mat& output, const bool normalize, const bool exchangeRB)
{
	input.convertTo(output, CV_32F);
	if (normalize) {
		output = output / 255.0; // ��һ����[0, 1]
	}
	if (exchangeRB) {
		cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
	}
}

//ͼ��ǰ����
float preprocess(const cv::Mat& input_image, float input_tensor[], int width, int height, int channels)
{
	/// letterbox�任: ���ı��߱�(aspect ratio), ��input_image���Ų����õ�blob_image���Ͻ�

	// ��������
	const float scale = std::min(height / float(input_image.rows), width / float(input_image.cols));
	const cv::Matx23f matrix{
		scale, 0.0, 0.0,
		0.0, scale, 0.0,
	};
	cv::Mat blob_image;
	// �������scale��Χ��������ת��, ��ֻ��Ϊ�����һ���ٶ�(��Ҫ������˽���ͨ�����ٶ�), ���ɶ��Ժܲ�
	// �������������ٶ������Ŀ��Թ̶�һ������(ǰ����if��֧������)
	if (scale > 1.0 + FLT_EPSILON) {
		// Ҫ�Ŵ�, ��ô�Ƚ���ͨ���ٷŴ�
		convert(input_image, blob_image, true, true);
		cv::warpAffine(blob_image, blob_image, matrix, cv::Size(width, height));
	}
	else if (scale < 1.0 - FLT_EPSILON) {
		// Ҫ��С, ��ô����С�ٽ���ͨ��
		cv::warpAffine(input_image, blob_image, matrix, cv::Size(width, height));
		convert(blob_image, blob_image, true, true);
	}
	else {
		convert(input_image, blob_image, true, true);
	}

	/// ��ͼ����������input_tensor
	float* const input_tensor_data = (float*)input_tensor;
	// ԭ��ͼƬ����Ϊ HWC��ʽ��ģ������ڵ�Ҫ���Ϊ CHW ��ʽ
	for (size_t c = 0; c < channels; c++) {
		for (size_t h = 0; h < height; h++) {
			for (size_t w = 0; w < width; w++) {
				input_tensor_data[c * width * height + h * width + w] = blob_image.at<cv::Vec<float, 3>>(h, w)[c];
			}
		}
	}
	return 1 / scale;
}


//ǰ����
void preprocess(cv::Mat& img, float data[]) {
	int w, h, x, y;
	float r_w = INPUT_W / (img.cols*1.0);
	float r_h = INPUT_H / (img.rows*1.0);
	if (r_h > r_w) {
		w = INPUT_W;
		h = r_w * img.rows;
		x = 0;
		y = (INPUT_H - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = INPUT_H;
		x = (INPUT_W - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
	//cudaResize(img, re);
	cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

	int i = 0;
	for (int row = 0; row < INPUT_H; ++row) {
		uchar* uc_pixel = out.data + row * out.step;
		for (int col = 0; col < INPUT_W; ++col) {
			data[i] = (float)uc_pixel[2] / 255.0;
			data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
			data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;
		}
	}
}


//ֻ���boxӳ���ԭͼ
std::vector<Bbox> rescale_box(std::vector<Bbox> &out, int width, int height) {
	float gain = 640.0 / std::max(width, height);
	float pad_x = (640.0 - width * gain) / 2;
	float pad_y = (640.0 - height * gain) / 2;

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

//���ӻ�
cv::Mat renderBoundingBox(cv::Mat image, const std::vector<Bbox> &bboxes) {
	for (const auto &rect : bboxes)
	{

		cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
		cv::rectangle(image, rst, cv::Scalar(255, 204, 0), 2, cv::LINE_8, 0);
		//cv::rectangle(image, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2), cv::Point(rect.x + rect.w / 2, rect.y + rect.h / 2), cv::Scalar(255, 204,0), 3);

		int baseLine;
		std::string label = class_names[rect.classes] + ": " + std::to_string(rect.score * 100).substr(0, 4) + "%";

		cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_TRIPLEX, 0.5, 1, &baseLine);
		//int newY = std::max(rect.y, labelSize.height);
		rectangle(image, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - round(1.5*labelSize.height)),
			cv::Point(rect.x - rect.w / 2 + round(1.0*labelSize.width), rect.y - rect.h / 2 + baseLine), cv::Scalar(255, 204, 0), cv::FILLED);
		cv::putText(image, label, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 204, 255));


	}
	return image;
}

float h_input[INPUT_SIZE * INPUT_SIZE * 3];
float h_output[OUTPUT_SIZE*(NUMS_CLASS+4)];   //1
float kNmsThresh;
float kConfThresh;
float kClassScore;
bool  afterScale = false;
float factor;
int main() 
{
	Logger gLogger;
	//��ʼ�����������plugin�����ʼ��plugin respo
    nvinfer1:initLibNvInferPlugins(&gLogger, "");


	nvinfer1::IRuntime* engine_runtime = nvinfer1::createInferRuntime(gLogger);
	
	//D:\\AI\\YOLO\\yolov8-main\\runs\\detect\\train3\\weights\\best_1_nms_fp16.engine
    // D:\\AI\\YOLO\\ultralytics-main\\runs\\detect\\train6\\weights\\best_1_nms_fp16.engine
	// 
	//D:\\trtOnnx\\good\\best_1_nms_fp16.engine
    //D:\\trtOnnx\\nogood\\best_1_nms_fp16.engine
	std::string engine_filepath = "D:\\AI\\ObjectDetector\\YOLO\\ultralytics-main\\runs\\detect\\train3\\weights\\best_fp16.engine";

	std::ifstream file;
	file.open(engine_filepath, std::ios::binary | std::ios::in);
	file.seekg(0, std::ios::end);
	int length = file.tellg();
	file.seekg(0, std::ios::beg);

	std::shared_ptr<char> data(new char[length], std::default_delete<char[]>());
	file.read(data.get(), length);
	file.close();

	//nvinfer1::ICudaEngine* engine_infer = engine_runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
	nvinfer1::ICudaEngine* engine_infer = engine_runtime->deserializeCudaEngine(data.get(), length, nullptr);
	nvinfer1::IExecutionContext* engine_context = engine_infer->createExecutionContext();

	int input_index = engine_infer->getBindingIndex("images"); //1x3x640x640
	//std::string input_name = engine_infer->getBindingName(0)
	int output_index = engine_infer->getBindingIndex("output0");  //1

	nvinfer1::Dims inputSize=engine_infer->getBindingDimensions(input_index);
	nvinfer1::Dims outputSize = engine_infer->getBindingDimensions(output_index);

	std::cout << "�����index: " << input_index << " �����num_detections-> " << output_index  << std::endl;

	if (engine_context == nullptr)
	{
		std::cerr << "Failed to create TensorRT Execution Context." << std::endl;
	}

	// cached_engine->destroy();
	std::cout << "loaded trt model , do inference" << std::endl;


	cv::String image_dir = "E:\\DataSets\\keypoint\\images\\testImages";
	std::vector<cv::String> fn;
	cv::glob(image_dir, fn, false);
	std::vector<cv::Mat> images;
	size_t count = fn.size(); //number of png files in images folde

	std::cout << count << std::endl;
	int nums = 0;
	for (size_t i = 0; i < count; i++)
	{
		cv::Mat image = cv::imread(fn[i]);
		cv::Mat disImage = image.clone();

		////cv2��ͼƬ
		//cv::Mat image;
		//image = cv::imread("./test.jpg", 1);
		std::cout << fn[i] << std::endl;
		afterScale = true;
		//preProcess
		if (afterScale )
		{
			//����1��
			preprocess(image, h_input);
		}
		else
		{
			//����2��
		    factor = preprocess(image, h_input, INPUT_W, INPUT_H, 3);
		}

		void* buffers[2];
		cudaMalloc(&buffers[0], INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));  //<- input
		cudaMalloc(&buffers[1], OUTPUT_SIZE*(NUMS_CLASS+4) * sizeof(float)); //<- num_detections

		cudaMemcpy(buffers[0], h_input, INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float), cudaMemcpyHostToDevice);

		// -- do execute --------//
		auto infer_begintime = cv::getTickCount();
		engine_context->executeV2(buffers);
		auto infer_endtime = cv::getTickCount();
		auto infer_time = (to_string)((infer_endtime - infer_begintime) * 1000 / getTickFrequency());
		cout << "trt����ʱ��ms:" << infer_time << endl;

		cudaMemcpy(h_output, buffers[1], OUTPUT_SIZE * (NUMS_CLASS + 4) * sizeof(float), cudaMemcpyDeviceToHost);

		//postProcess    
		const int out_rows = NUMS_CLASS+4; //���"output"�ڵ��rows
		const int out_cols = OUTPUT_SIZE; //���"output"�ڵ��cols
		const cv::Mat det_output(out_rows, out_cols, CV_32F, (float*)h_output);

		std::vector<cv::Rect> boxes;
		std::vector<int> class_ids;
		std::vector<float> confidences;
		kNmsThresh = 0.3f;
		kConfThresh = 0.2f;
		kClassScore = 0.2f;

		//����1��ֱ�ӵõ�ԭͼ��bbox�ߴ�
		// �����ʽ��[11,8400], ÿ�д���һ����(�������8400����), ǰ��4�зֱ���cx, cy, ow, oh, ����7����ÿ���������Ŷ�
		for (int i = 0; i < det_output.cols; ++i) {
			const cv::Mat classes_scores = det_output.col(i).rowRange(4, 11);//�����÷�ȡ����
			cv::Point class_id_point;
			double score;
			cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);//�ҵ���Ӧ�÷��������������

			// ���Ŷ� 0��1֮��
			if (score > kClassScore) {
				const float cx = det_output.at<float>(0, i);
				const float cy = det_output.at<float>(1, i);
				const float ow = det_output.at<float>(2, i);
				const float oh = det_output.at<float>(3, i);
				cv::Rect box;
				if (afterScale)
				{
					box.x = static_cast<int>(cx );
					box.y = static_cast<int>(cy );
					box.width = static_cast<int>(ow );
					box.height = static_cast<int>(oh );
				}
				else
				{
					//const float scale = std::min(INPUT_H / float(image.rows), INPUT_W / float(image.cols));
					//const float factor = 1 / scale;
					box.x = static_cast<int>((cx - 0.5 * ow) * factor);
					box.y = static_cast<int>((cy - 0.5 * oh) * factor);
					box.width = static_cast<int>(ow * factor);
					box.height = static_cast<int>(oh * factor);
				}
				boxes.push_back(box);
				class_ids.push_back(class_id_point.y);//class_id_point=point(i,class),class�Ƕ�Ӧ���������point.y
				confidences.push_back(score);
			}
		}

		// NMS, �������нϵ����Ŷȵ������ص���
		std::vector<int> indexes;
		cv::dnn::NMSBoxes(boxes, confidences, kConfThresh, kNmsThresh, indexes);
		if (!afterScale)
		{
			//����1:
			for (size_t i = 0; i < indexes.size(); i++) {
				const int index = indexes[i];
				const int idx = class_ids[index];
				cv::rectangle(disImage, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
				cv::rectangle(disImage, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
					cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
				string nameScore = class_names[idx] + "  " + std::to_string(confidences[idx]);
				cv::putText(disImage, nameScore, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			}

			std::string savePath = "trt_res/result" + std::to_string(nums) + ".jpg";
			cv::imwrite(savePath, disImage);
		}
		else
		{
			//����2���õ�ģ�������bbox�ߴ磬��ת��Ϊԭͼ
			std::vector<Bbox> pred_box;
			//����1:
			for (size_t i = 0; i < indexes.size(); i++) {
				const int index = indexes[i];
				const int idx = class_ids[index];
				
				Bbox box;				
				box.x = boxes[index].x;  //(h_output_1[i * 4 + 2] + h_output_1[i * 4]) / 2.0;
				box.y = boxes[index].y;// (h_output_1[i * 4 + 3] + h_output_1[i * 4 + 1]) / 2.0;
				box.w = boxes[index].width;// h_output_1[i * 4 + 2] - h_output_1[i * 4];
				box.h = boxes[index].height; // h_output_1[i * 4 + 3] - h_output_1[i * 4 + 1];
				box.score = confidences[idx];
				box.classes = (int)class_ids[index];

				pred_box.push_back(box);				
			}
	
			std::vector<Bbox> out = rescale_box(pred_box, image.cols, image.rows);

			cv::Mat img = renderBoundingBox(image, out);
			std::string savePath = "trt_res/result" + std::to_string(nums) + ".jpg";
			cv::imwrite(savePath, img);
			nums++;
		}

		cudaFree(buffers[0]);
		cudaFree(buffers[1]);
	}

	engine_runtime->destroy();
	engine_infer->destroy();

	return 0;
}