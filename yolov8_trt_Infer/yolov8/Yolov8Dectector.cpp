#include "Yolov8Dectector.h"



Yolov8Dectector::~Yolov8Dectector()
{
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

Logger logger;
/// <summary>
/// 将onnx生成engine模型
/// </summary>
/// <param name="onnx_path">onnx_path</param>
/// <param name="format">"FP16"or"INT8"</param>
void Yolov8Dectector::Onnx2Engine(string onnx_path,string engine_path,string format="FP16")
{

	IBuilder* builder = createInferBuilder(logger);
    assert(builder);
	INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)); //是0U还是1u需视情况而定
    assert(network);

    IParser* parser = createParser(*network, logger);
    assert(parser);

	parser->parseFromFile(onnx_path.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));

	for (int32_t i = 0; i < parser->getNbErrors(); ++i)
	{
		std::cout << parser->getError(i)->desc() << std::endl;
	}
	std::cout << "successfully parse the onnx model" << std::endl;

    //创建config
    IBuilderConfig* config = builder->createBuilderConfig();
    assert(config);

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(16*(1 << 20));

    //创建profile
    //auto profile = builder->createOptimizationProfile();
    //auto input_tensor = network->getInput(0);
    //auto input_dims = input_tensor->getDimensions();

    //profile->setDimensions(input_tensor->getName(), OptProfileSelector::kMIN, Dims4(1, 3, 640, 640)); // 设置输入x的动态维度，最小值
    //profile->setDimensions(input_tensor->getName(), OptProfileSelector::kOPT, Dims4(2, 3, 640, 640)); // 期望输入的最优值
    //profile->setDimensions(input_tensor->getName(), OptProfileSelector::kMAX, Dims4(4, 3, 640, 640)); // 最大值


    ////input_dims.d[0] = 1;
    ////profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    ////profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
    ////input_dims.d[0] = kYolov8Param.kBatchSize;
    ////profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);

    //config->addOptimizationProfile(profile);

    if(format=="FP16")
        config->setFlag(BuilderFlag::kFP16);
    else if (format == "INT8")
    {
        config->setFlag(BuilderFlag::kINT8);
    }

    //创建engine
    //ofstream modelStream;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine);
    IHostMemory* serialized_engine = engine->serialize();
    assert(serialized_engine);
 
    std::ofstream p(engine_path, std::ios::binary);


    if (!p)
    {
        std::cerr << "could not open plan output file" << std::endl;
        assert(false);
        //return -1;
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
   
    serialized_engine->destroy();
    engine->destroy();
    network->destroy();
    parser->destroy();
    builder->destroy();

}

void Yolov8Dectector::deserialize_engine(std::string& engine_name)
{
    //读取
    //判断engine是否存在，存在直接加载engine，不存在调用onnxpaser构建获取engine
    std::ifstream file;
    int length;
    file.open(engine_name, std::ios::binary | std::ios::in);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    file.seekg(0, std::ios::end);
    length = file.tellg();
    file.seekg(0, std::ios::beg);

    std::shared_ptr<char> data(new char[length], std::default_delete<char[]>());
    file.read(data.get(), length);
    file.close();

     runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime);
    engine = runtime->deserializeCudaEngine(data.get(), length, nullptr);
    assert(engine);

     context = engine->createExecutionContext();
    assert(context);
}

void Yolov8Dectector::setInputBindings(ICudaEngine* engine_infer, IExecutionContext* engine_context)
{
    assert(engine_infer->getNbBindings() == 2);
    int input_index = engine_infer->getBindingIndex(kYolov8Param.inputName.data()); //1x3x640x640
    int output_index = engine_infer->getBindingIndex(kYolov8Param.outputName.data());  //1
    assert(input_index == 0);
    assert(output_index == 1);

    //engine模型动态batch（BATCH_SIZE, 3, width, height）
    nvinfer1::Dims inputSize = engine_infer->getBindingDimensions(input_index);
    nvinfer1::Dims outputSize = engine_infer->getBindingDimensions(output_index);

    std::cout << "输入的index: " << input_index << " 输出的num_detections-> " << output_index << std::endl;
    // 固定context的输入为（BATCH_SIZE, 3, 640, 640）
    engine_context->setBindingDimensions(0, nvinfer1::Dims4(kYolov8Param.kBatchSize, kYolov8Param.kChannel, kYolov8Param.kInputW, kYolov8Param.kInputH));
}

/// <summary>
/// 初始化engine,生成context
/// </summary>
/// <param name="engine_name">EnginePath</param>
void  Yolov8Dectector::InitialModel(std::string& engine_name)
{
    if (false)
    {
        std::ifstream file;
        int length;
        std::shared_ptr<char> data;
        file.open(engine_name, std::ios::binary | std::ios::in);
        if (!file.good()) {
            std::cerr << " engine fils-- " << engine_name << " is not exits!" << std::endl;
            string onnx_path = engine_name.substr(0, engine_name.find_last_of(".")) + ".onnx";
            Onnx2Engine(onnx_path, engine_name, "FP16");
        }
    }


    deserialize_engine(engine_name);
    setInputBindings(engine,context);

    cudaStreamCreate(&stream);
    outputSize= kYolov8Param.kBatchSize * kYolov8Param.kOutputSize * (kYolov8Param.kClassNums + 4);
    inputSize = kYolov8Param.kBatchSize * kYolov8Param.kChannel * kYolov8Param.kInputW * kYolov8Param.kInputH;
}
/// <summary>
/// 
/// </summary>
/// <param name="input_buffer_host">batchImageInput</param>
/// <param name="nums_out">numsBoxePerImage</param>
/// <param name="boxes_out">boxes_out</param>
/// <param name="class_ids_out">class_ids_out</param>
/// <param name="confidences_out">confidences_out</param>
/// <param name="org_width">orgImageWidth</param>
/// <param name="org_height">orgImageHeight</param>
void  Yolov8Dectector::Infer( uchar* input_buffer_host, int* nums_out, int* boxes_out, int* class_ids_out, float* confidences_out, int org_width, int org_height)
{
    //处理输入图片的数据
    float* h_inputs = new float[inputSize];
    int step = kYolov8Param.kInputW* kYolov8Param.kInputH* kYolov8Param.kChannel;
    for (size_t bs_i = 0; bs_i < kYolov8Param.kBatchSize; bs_i++)
    {
        Mat inputImage(org_height,org_width,CV_8UC3);
        memcpy(inputImage.data, input_buffer_host + bs_i * org_width * org_height * 3, org_width * org_height * 3 * sizeof(uchar));
        
       //imwrite(std::to_string(bs_i)+ ".jpg",inputImage);

        float* h_input_new = new float[step];
        preprocess(inputImage, h_input_new, kYolov8Param.kInputW, kYolov8Param.kInputH, kYolov8Param.kChannel);

        memcpy(h_inputs + bs_i*step, h_input_new, step * sizeof(float));

        delete[] h_input_new;
    }


    //输入数据传输到设备
    float* out_buffer_host = new float[outputSize];
    void* buffers[2];
    cudaMalloc((void**)&buffers[0], inputSize * sizeof(float));
    cudaMalloc((void**)&buffers[1], outputSize * sizeof(float));
    cudaMemcpy(buffers[0], h_inputs,  inputSize * sizeof(float), cudaMemcpyHostToDevice);
   
    //float* out_buffer_host = new float[kYolov8Param.kBatchSize * kYolov8Param.kOutputSize * sizeof(float)];
    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start,0);
    auto infer_begintime = std::chrono::system_clock::now();
    
    //同步推理
    //context->executeV2(buffers);

    //异步推理
    //context->enqueue(batchSize,buffers, stream, nullptr);
    context->enqueueV2(buffers, stream, nullptr);
    cudaDeviceSynchronize();

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time,start,end);
    cout << "trt异步推理时间ms:" << time << endl;
    auto infer_endtime = std::chrono::system_clock::now();
    auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(infer_endtime - infer_begintime).count();
    //cout << "trt推理时间ms:" << infer_time << endl;

    //将设备结果拷出到主机
    cudaMemcpyAsync(out_buffer_host, buffers[1],outputSize*sizeof(float),cudaMemcpyDeviceToHost,stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    //处理最终结果
    getResult(out_buffer_host, boxes_out, class_ids_out, confidences_out, nums_out,org_width,org_height);

    delete[] out_buffer_host;
}

/// <summary>
/// 
/// </summary>
/// <param name="out_buffer_host">inferResult</param>
/// <param name="boxes_out">boxesOut</param>
/// <param name="class_ids_out">idsOut</param>
/// <param name="confidences_out">confOut</param>
/// <param name="nums_out">boxNumsImagePer</param>
/// <param name="org_width">orgImageWidth</param>
/// <param name="org_height">orgImageHeight</param>
void Yolov8Dectector::getResult(float* out_buffer_host, int* boxes_out, int* class_ids_out, float* confidences_out, int* nums_out,int org_width,int org_height)
{
    std::vector<int> boxes_res;
    std::vector<int> class_ids_res;
    std::vector<float> confidences_res;
    int nums = 0;
    for (size_t bsi = 0; bsi < kYolov8Param.kBatchSize; bsi++)
    {
        int  step = bsi * (outputSize/ kYolov8Param.kBatchSize);

        const int out_rows = kYolov8Param.kClassNums + 4; //获得"output"节点的rows
        const int out_cols = kYolov8Param.kOutputSize; //获得"output"节点的cols
        const cv::Mat det_output(out_rows, out_cols, CV_32F, (float*)out_buffer_host + step);

        std::vector<cv::Rect> boxes;
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<int> boxes_int;
        float kClassScore = 0.2f;

        //解析结果
        // 
        //方法：直接得到原图的bbox尺寸
        // 输出格式是[11,8400], 每列代表一个框(即最多有8400个框), 前面4行分别是cx, cy, ow, oh, 后面7行是每个类别的置信度
        for (int i = 0; i < det_output.cols; ++i) {
            const cv::Mat classes_scores = det_output.col(i).rowRange(4, 11);//将类别得分取出来
            cv::Point class_id_point;
            double score;
            cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);//找到对应得分最大的类别及其坐标

            // 置信度 0～1之间
            if (score > kClassScore) {
                const float cx = det_output.at<float>(0, i);
                const float cy = det_output.at<float>(1, i);
                const float ow = det_output.at<float>(2, i);
                const float oh = det_output.at<float>(3, i);
                cv::Rect box;
                box.x = static_cast<int>(cx);
                box.y = static_cast<int>(cy);
                box.width = static_cast<int>(ow);
                box.height = static_cast<int>(oh);
                
                boxes.push_back(box);
                class_ids.push_back(class_id_point.y);//class_id_point=point(i,class),class是对应的类别，属于point.y
                confidences.push_back(score);
            }
        }

        // NMS, 消除具有较低置信度的冗余重叠框
        std::vector<int> indexes;
        cv::dnn::NMSBoxes(boxes, confidences, kYolov8Param.kConfThresh, kYolov8Param.kNmsThresh, indexes);
        
        //将box还原为原图尺寸
        std::vector < cv::Rect > out_boxes = rescale_box(boxes, org_width, org_height);
        nums_out[bsi] = indexes.size();
        nums += nums_out[bsi];

        //处理最终输出结果，全部以指针形式输出
        for (size_t i = 0; i < nums_out[bsi]; i++)
        {
            int index = indexes[i];
            boxes_res.push_back((int)out_boxes[index].x);
            boxes_res.push_back((int)out_boxes[index].y);
            boxes_res.push_back((int)out_boxes[index].width);
            boxes_res.push_back((int)out_boxes[index].height);
            class_ids_res.push_back(class_ids[index]);
            confidences_res.push_back(confidences[index]);
        }    
    }

    std::memcpy(boxes_out, boxes_res.data(), 4 * nums * sizeof(int));
    std::memcpy(class_ids_out, class_ids_res.data(), nums * sizeof(int));
    std::memcpy(confidences_out, confidences_res.data(), nums * sizeof(float));
}


