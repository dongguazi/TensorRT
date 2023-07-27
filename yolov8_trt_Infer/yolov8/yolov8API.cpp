

#include"Yolov8Dectector.h"

extern "C" Yolov8Dectector * CreateDetector_GPU()
{
	Yolov8Dectector* m_yolov8 = new Yolov8Dectector();
	return m_yolov8;
}

extern "C" void InitializeDetector_GPU(Yolov8Dectector * m_yolov8, const char* engine_path_char,bool Onnx2Trt)
{
	//stringÀàÐÍ×ª»»
	string engine_path(engine_path_char);
	m_yolov8->InitialModel(engine_path, Onnx2Trt);
}

extern "C" void PredictDetector_CPU(Yolov8Dectector * m_yolov8,
	uchar* input_buffer_host, int* nums_out, int* boxes_out, int* class_ids_out, float* confidences_out,int width,int height)
{
	m_yolov8->Infer( input_buffer_host,  nums_out,  boxes_out, class_ids_out, confidences_out, width, height);
}

extern "C"  void DisposeDetector_CPU(Yolov8Dectector * m_yolov8)
{
	delete m_yolov8;
}