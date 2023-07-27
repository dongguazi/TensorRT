#include "preprocess.h"



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
float preprocessWithRatio(const cv::Mat& input_image, float input_tensor[], int width, int height, int channels)
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
void preprocess(cv::Mat& img, float* data, int model_width, int model_height, int model_channels) {
	int w, h, x, y;
	int INPUT_W = model_width, INPUT_H = model_height;
	float r_w = INPUT_W / (img.cols * 1.0);
	float r_h = INPUT_H / (img.rows * 1.0);
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
