### yolov8动态batch的TensorRT推理
--1. 接口封装，简单易用 
    将yolov8的infer过程封装为dll文件，对外提供了5个API接口:
    大大简化了API调用者的难度，不同的语言只要支持dll的都可以调用;
    另外提供Onnx2Engine接口可在新设备上将onnx转为engine模型，infer和转换模型逻辑交给调用者决定。

    
    extern "C" Yolov8Dectector * CreateDetector_GPU()
    {
        Yolov8Dectector* m_yolov8 = new Yolov8Dectector();
        return m_yolov8;
    }

    extern "C" void InitializeDetector_GPU(Yolov8Dectector * m_yolov8, const char* engine_path_char)
    {
        //string类型转换
        string engine_path(engine_path_char);
        m_yolov8->InitialModel(engine_path);
    }

    extern "C" void PredictDetector_CPU(Yolov8Dectector * m_yolov8,
        uchar* input_buffer_host, int* nums_out, int* boxes_out, int* class_ids_out, float* confidences_out,int width,int height)
    {
        m_yolov8->Infer( input_buffer_host,  nums_out,  boxes_out, class_ids_out, confidences_out, width, height);
    }

    extern "C" Yolov8Dectector * Onnx2Engine(Yolov8Dectector * m_yolov8, const char* onnx_path, const char*engine_path, const char* format )
    {
        string onnxPath(onnx_path);
        string enginePath(engine_path);
        string format_(format);
        m_yolov8->Onnx2Engine(onnxPath, enginePath, format_);
    }

    extern "C"  void DisposeDetector_GPU(Yolov8Dectector * m_yolov8)
    {
        delete m_yolov8;
    }


--2. 支持多batch推理,所有参数可以更改。
    通过调用Yolov8Dectector类中的YoloV8Param对参数进行更改，针对不同的yolov8的模型结构，可以通过指定inputName和outputName，以便定位模型的输入和输出。
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
      
--3. demo 
    项目中yolov8_trt_infer.cpp提供了对原始函数推理过程，仅供参考，不会影响实际API的使用。

--4. 推理性能
     输入图片大小3072*2048，模型输入大小640*640，batchsize=6，一个batch推理时间6ms+，平均一张图的推理时间再1ms左右。
     推理结果：
     trt异步推理时间ms:6.7185
     trt异步推理时间ms:6.6503
     trt异步推理时间ms:6.66579
     trt异步推理时间ms:6.63389
     trt异步推理时间ms:6.67002
