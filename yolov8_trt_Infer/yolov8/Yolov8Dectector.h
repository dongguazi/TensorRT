#pragma once
#include "NvOnnxParser.h"
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

#include "preprocess.h"
#include "postprocess.h"

#include <npp.h>
#include "logging.h"

using namespace sample;
using namespace std;
using namespace cv;
using namespace nvinfer1;
using namespace nvonnxparser;


class Yolov8Dectector
{
public: Yolov8Dectector()=default;
public: ~Yolov8Dectector();

	  struct YoloV8Param
	  {
		  int kBatchSize = 1;
		  int kMaxBatchSize = 1;
		  int kInputW = 640;
		  int kInputH = 640;
		  int kChannel = 3;
		  int kOutputSize=8400;
		  float kNmsThresh = 0.3f;
		  float kConfThresh = 0.2f;
		  int kClassNums = 7;
		  string inputName = "images";
		  string outputName = "output0";
	  };
	  YoloV8Param kYolov8Param;
private:
	  IRuntime* runtime = nullptr;
	  ICudaEngine* engine = nullptr;
	  IExecutionContext* context = nullptr;
	  cudaStream_t stream;

	  int outputSize;
	  int inputSize;

	  //float* input_buffers;
	  //float* output_buffers;

	  Logger gLogger;

public:
	  void InitialModel(std::string& engine_name, bool Onnx2Trt);
	  void Infer(uchar* input_buffer_host, int* nums_out, int* boxes_out, int* class_ids_out, float* confidences_out, int width, int height);
	  void Onnx2Engine(string onnx_path, string engine_path, string format );

private:
	  void deserialize_engine(std::string& engine_name);
	  void setInputBindings(ICudaEngine* engine_infer, IExecutionContext* engine_context);
	  void getResult(float* out_buffer_host, int* boxes_out, int* class_ids_out, float* confidences_out, int* nums_out, int width, int height);

};

