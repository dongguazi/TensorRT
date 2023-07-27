// xujing
//YOLOv8

# include "Yolov8Dectector.h"

#define BATCH_SIZE 6
#define INPUT_W 640
#define INPUT_H 640
#define INPUT_SIZE 640
#define INPUT_CH 3


#define NUMS_CLASS 7
#define OUTPUT_SIZE 8400*(NUMS_CLASS+4)

#define IsPadding 1


std::vector<std::string> class_names = { "rubber stopper", "push rod tail", "needle tail", "mouth", "crooked mouth", "screw mouth", "small rubber plug" };


float h_input[INPUT_SIZE * INPUT_SIZE * 3];


float h_output[OUTPUT_SIZE*(NUMS_CLASS+4)* BATCH_SIZE];   //1
float kNmsThresh;
float kConfThresh;
float kClassScore;
bool  afterScale = false;
float factor;
Yolov8Dectector* yolo;


int main() 
{



	std::string engine_filepath = "D:\\AI\\ObjectDetector\\YOLO\\ultralytics-main\\runs\\detect\\train3\\weights\\best_fp16_dynamic.engine";

	yolo=new Yolov8Dectector();
	yolo->kYolov8Param.kBatchSize = BATCH_SIZE;
	yolo->kYolov8Param.kMaxBatchSize = BATCH_SIZE+1;

	//yolo->Onnx2Engine("D:\\AI\\Infers\\TRT\\yolov8_trt_Infer\\yolov8\\best.onnx","D:\\AI\\Infers\\TRT\\yolov8_trt_Infer\\yolov8\\best.engine","FP16");


	yolo->InitialModel(engine_filepath);


	//------准备输入
	
	cv::String image_dir = "E:\\DataSets\\keypoint\\images\\valImages";
	std::vector<cv::String> fn;
	cv::glob(image_dir, fn, false);

	size_t count = fn.size(); //number of png files in images folde

	std::cout << count << std::endl;
	int nums = 0;
	vector <cv::Mat> images;
	vector <cv::Mat> disPlayImages;
	int  org_width;
	int  org_height;
	for (size_t i = 0; i < count; i++)
	{
		cv::Mat image = cv::imread(fn[i]);
		cv::Mat disImage = image.clone();
		org_width = image.cols;
		org_height = image.rows;
		images.push_back(image);
		disPlayImages.push_back(disImage);
	}

	int  cycleNums = images.size() / BATCH_SIZE;


	for (size_t cycleNum = 0; cycleNum < cycleNums; cycleNum++)
	{
		//vector<Mat> batchImage;
		//for (size_t bs_i = 0; bs_i < BATCH_SIZE; bs_i++)
		//{
		//	///cv2读图片
		//	cv::Mat image = images[cycleNum*BATCH_SIZE+bs_i];
		//	std::cout << fn[cycleNum] << std::endl;
		//	org_width = image.cols;
		//	org_height = image.rows;
		//	batchImage.push_back(image);
		//}

		int  stepPerImage = org_width * org_height * 3;
		int  sizesAllImage = stepPerImage * BATCH_SIZE;
		uchar* h_inputs = new unsigned char[sizesAllImage];

		unsigned char* img_batch = new unsigned char[sizesAllImage];

		//memcpy(img_batch , images.data()+cycleNum * BATCH_SIZE , sizesAllImage * sizeof(unsigned char));

		//将batch中每张图片输入到图片指针0,1,2
		for (size_t batch_size_i = 0; batch_size_i < BATCH_SIZE; batch_size_i++)
		{
			//将所有batch的图片加载到图片指针上
			memcpy(img_batch + batch_size_i * stepPerImage, images[cycleNum * BATCH_SIZE + batch_size_i].data, stepPerImage * sizeof(unsigned char));
		}

		//preProcess

		int* box_nums = new int[BATCH_SIZE];
		int* boxes = new int[10000];
		int * ids= new int[10000];
		float* conf = new float[10000];


		yolo->Infer(img_batch, box_nums, boxes, ids, conf, org_width, org_height);

		delete[]h_inputs;
		//postProcess    
		int totalNums = 0;
		for (size_t bs_i = 0; bs_i < BATCH_SIZE; bs_i++)
		{
			Mat disImage = disPlayImages[cycleNum * BATCH_SIZE + bs_i];
			
			//方法2：得到模型输出的bbox尺寸，再转换为原图
			std::vector<Bbox> pred_box;
			int  box_num = box_nums[bs_i];

			for (size_t i = 0; i < box_num; i++)
			{
				cv::Rect2i rect(boxes[totalNums * 4 + 4 * i + 0], boxes[totalNums * 4 + 4 * i + 1], boxes[totalNums * 4 + 4 * i + 2], boxes[totalNums * 4 + 4 * i + 3]);
				Bbox box;
				box.x = rect.x;  //(h_output_1[i * 4 + 2] + h_output_1[i * 4]) / 2.0;
				box.y = rect.y;// (h_output_1[i * 4 + 3] + h_output_1[i * 4 + 1]) / 2.0;
				box.w = rect.width;// h_output_1[i * 4 + 2] - h_output_1[i * 4];
				box.h = rect.height; // h_output_1[i * 4 + 3] - h_output_1[i * 4 + 1];
				box.score = conf[totalNums + i];
				box.classes = ids[totalNums + i];

				pred_box.push_back(box);

			}

			//std::vector<Bbox> out = rescale_box(pred_box, disImage.cols, disImage.rows);

			cv::Mat img = renderBoundingBox(disImage, pred_box, class_names);
			std::string savePath = "trt_res/result_" + std::to_string(cycleNum) + "_" + std::to_string(bs_i) + ".jpg";
			cv::imwrite(savePath, img);
			totalNums += box_nums[bs_i];
		}

	}

	//yolo->~Yolov8Dectector();
	delete yolo;

	return 0;
}