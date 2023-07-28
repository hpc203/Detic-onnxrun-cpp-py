#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>  ///nvidia-cuda加速
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

typedef struct BoxInfo
{
	int xmin;
	int ymin;
	int xmax;
	int ymax;
	float score;
	string name;
} BoxInfo;

class Detic
{
public:
	Detic(string modelpath);
	vector<BoxInfo> detect(Mat cv_image);
private:
	void preprocess(Mat srcimg);
	vector<float> input_image_;
	int inpWidth;
	int inpHeight;
	vector<string> class_names;
	const int max_size = 800;

	//存储初始化获得的可执行网络
	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Head Pose Estimation");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

Detic::Detic(string model_path)
{
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);  ///nvidia-cuda加速
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());   ///如果在windows系统就这么写
	ort_session = new Session(env, widestr.c_str(), sessionOptions);   ///如果在windows系统就这么写
	///ort_session = new Session(env, model_path.c_str(), sessionOptions);  ///如果在linux系统，就这么写

	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}

	ifstream ifs("imagenet_21k_class_names.txt");
	string line;
	while (getline(ifs, line))
	{
		this->class_names.push_back(line);  ///你可以用随机数给每个类别分配RGB值
	}
}

void Detic::preprocess(Mat srcimg)
{
	Mat dstimg;
	cvtColor(srcimg, dstimg, COLOR_BGR2RGB);
	int im_h = srcimg.rows;
	int im_w = srcimg.cols;
	float oh, ow, scale;
	if (im_h < im_w)
	{
		scale = (float)max_size / (float)im_h;
		oh = max_size;
		ow = scale * (float)im_w;
	}
	else
	{
		scale = (float)max_size / (float)im_h;
		oh = scale * (float)im_h;
		ow = max_size;
	}
	float max_hw = std::max(oh, ow);
	if (max_hw > max_size)
	{
		scale = (float)max_size / max_hw;
		oh *= scale;
		ow *= scale;
	}

	resize(dstimg, dstimg, Size(int(ow + 0.5), int(oh + 0.5)), INTER_LINEAR);
	this->inpHeight = dstimg.rows;
	this->inpWidth = dstimg.cols;
	this->input_image_.resize(this->inpWidth * this->inpHeight * dstimg.channels());
	int k = 0;
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < this->inpHeight; i++)
		{
			for (int j = 0; j < this->inpWidth; j++)
			{
				float pix = dstimg.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[k] = pix;
				k++;
			}
		}
	}
}

vector<BoxInfo> Detic::detect(Mat srcimg)
{
	int im_h = srcimg.rows;
	int im_w = srcimg.cols;
	this->preprocess(srcimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());

	const float *pred_boxes = ort_outputs[0].GetTensorMutableData<float>();
	const float *scores = ort_outputs[1].GetTensorMutableData<float>();
	const int *pred_classes = ort_outputs[2].GetTensorMutableData<int>();
	//const float *pred_masks = ort_outputs[3].GetTensorMutableData<float>();

	int num_box = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[0];
	const float scale_x = float(im_w) / float(inpWidth);
	const float scale_y = float(im_h) / float(inpHeight);
	vector<BoxInfo> preds;
	for (int i = 0; i < num_box; i++)
	{
		float xmin = pred_boxes[i * 4] * scale_x;
		float ymin = pred_boxes[i * 4 + 1] * scale_y;
		float xmax = pred_boxes[i * 4 + 2] * scale_x;
		float ymax = pred_boxes[i * 4 + 3] * scale_y;
		xmin = std::min(std::max(xmin, 0.f), float(im_w));
		ymin = std::min(std::max(ymin, 0.f), float(im_h));
		xmax = std::min(std::max(xmax, 0.f), float(im_w));
		ymax = std::min(std::max(ymax, 0.f), float(im_h));

		const float threshold = 0;
		const float width = xmax - xmin;
		const float height = ymax - ymin;
		if (width > threshold && height > threshold)
		{
			preds.push_back({ int(xmin), int(ymin), int(xmax), int(ymax), scores[i], class_names[pred_classes[i]] });
		}
	}
	return preds;
}

int main()
{
	Detic mynet("weights/Detic_C2_R50_640_4x_in21k.onnx");
	string imgpath = "desk.jpg";
	Mat srcimg = imread(imgpath);
	vector<BoxInfo> preds = mynet.detect(srcimg);
	for (size_t i = 0; i < preds.size(); ++i)
	{
		rectangle(srcimg, Point(preds[i].xmin, preds[i].ymin), Point(preds[i].xmax, preds[i].ymax), Scalar(0, 0, 255), 2);
		string label = format("%.2f", preds[i].score);
		label = preds[i].name + " :" + label;
		putText(srcimg, label, Point(preds[i].xmin, preds[i].ymin - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
	}

	//imwrite("result.jpg", srcimg);
	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}