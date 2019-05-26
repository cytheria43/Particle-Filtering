#include <iostream>
#include<vector>
#include"partical.h"

Rect select;//选定区域。cv::Rect矩形类
vector<RotatedRect> select_rotate;
Point origin;//鼠标的点。opencv中提供的点的模板类
bool select_flag = false;//选择标志
bool tracking = false;//跟踪标志，用于判定是否已经选择了目标位置，只有它为true时，才开始跟踪
bool select_show = false;
Mat frame, hsv;//全局变量定义
int after_select_frames = 0;
vector<vector<Point>>contours;
vector<Vec4i>hierarchy;

string window_name = "select";

void detectContour(Mat &fr, Mat &result) {

	threshold(fr, fr, 45, 255, CV_THRESH_BINARY);
	//imshow("二值化", frame);
	Mat elem1 = getStructuringElement(MORPH_RECT, Size(2, 2));
	Mat elem2 = getStructuringElement(MORPH_RECT, Size(2, 2));
	erode(fr, fr, elem1);//腐蚀
	dilate(fr, fr, elem2);//膨胀

	findContours(fr, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//Mat dstImage = Mat::zeros(frame.size(), CV_8UC3);
	//for (int i = 0; i < hierarchy.size(); i++)
	//{
	//	Scalar color = Scalar(rand() % 255, rand() % 255, rand() % 255);
	//	drawContours(dstImage, contours, i, color, CV_FILLED, 8, hierarchy);
	//}
	//imshow("轮廓图", dstImage);
	vector<RotatedRect> box(contours.size());
	Point2f rect[4];
	for (int i = 0; i < contours.size(); i++)
	{
		box[i] = fitEllipse(Mat(contours[i]));
		RotatedRect rotate_t = minAreaRect(Mat(contours[i]));
		select_rotate.push_back(rotate_t);
		rotate_t.points(rect);
		cout << "box[i]====" << box[i].center << endl;
		for (int j = 0; j < 4; j++)
		{
			line(frame, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //绘制最小外接矩形每条边
		}

		//ellipse(frame, box[i], Scalar(0, 255, 0), 2, 8);
		//circle(frame, box[i].center, 3, Scalar(0, 0, 255), -1, 8);
		//select_ellipse(box[i].center.x, box[i].center.y, box[i].size.width, box[i].size.height, box[i].angle);
		select = box[i].boundingRect();
	}
	tracking = 1;
	select_show = 1;
	after_select_frames = 1;
}
void basicLinearTransformation(Mat &f) {
	double alpha = 0.9;   //1.0-3.0
	int beta = 5;  //0-100
	//namedWindow("亮度对比度");
	//createTrackbar("亮度", "亮度对比度", &beta, 100, on_trackbar1);
	//on_trackbar1(0, 0);
	Mat out = Mat::zeros(f.size(), f.type());
	for (int i = 0; i < f.rows; i++) {
		for (int j = 0; j < f.cols; j++) {
			for (int c = 0; c < 3; c++) {
				out.at<Vec3b>(i, j)[c] = alpha * f.at<Vec3b>(i, j)[c] + beta;
			}
		}
	}
	f = out;
}
void sharpen(Mat &img) {
	basicLinearTransformation(img);
	medianBlur(img, img, 5);
	//创建并初始化滤波模板
	Mat kernel(3, 3, CV_32F, cv::Scalar(0));
	kernel.at<float>(1, 1) = 8.0;
	kernel.at<float>(0, 1) = -1.0;
	kernel.at<float>(1, 0) = -1.0;
	kernel.at<float>(1, 2) = -1.0;
	kernel.at<float>(2, 1) = -1.0;
	//对图像进行滤波
	filter2D(img, img, img.depth(), kernel);
}



int main(int argc, unsigned char* argv[])
{
	char c;
	Mat target_img, track_img;
	Mat target_hist, track_hist;
	PARTICLE *pParticle;

	char filename[200];
	int i = 0;
	sprintf_s(filename, "D:\\VSproject\\video_detect\\video_detect\\mouse\\mouse1_%02d.jpg", i);
	frame = imread(filename); //上一帧
	/******通过对第一帧进行预处理得到粒子散布的旋转矩阵*********/
	Mat gg = frame.clone();
	sharpen(gg);
	cvtColor(gg, gg, CV_RGB2GRAY);
	bitwise_not(gg, gg);
	detectContour(gg, gg);
	imshow("selectrotate", frame);
	Rect select_copy = select;
	while (1)
	{
		/****读取一帧图像****/
		if (i == 100)
		{
			i = 0;
			select = select_copy;
			after_select_frames = 1;
		}
		sprintf_s(filename, "D:\\VSproject\\video_detect\\video_detect\\mouse\\mouse1_%02d.jpg", i++);
		frame = imread(filename);
		if (frame.empty())
			return -1;
		//sharpen(frame);
		/****将rgb空间转换为hsv空间****/
		cvtColor(frame, hsv, CV_BGR2HSV);

		if (tracking)
		{

			if (1 == after_select_frames)//选择完目标区域后
			{
				/****初始化目标粒子****/
				pParticle = particles;//指针初始化指向particles数组
				initializationOFpartical(hsv,select_rotate[0], target_hist, pParticle);
				
			}
			else if (2 == after_select_frames)//从第二帧开始就可以开始跟踪了
			{
				double sum = 0.0;
				pParticle = particles;
				RNG rng;//随机数产生器

				/****更新粒子结构体的大部分参数****/
				update_PARTICLES(hsv,pParticle, rng, track_hist, target_hist, sum);

				/****归一化粒子权重****/
				pParticle = particles;
				for (int i = 0; i < PARTICLE_NUMBER; i++)
				{
					pParticle->weight /= sum;
					pParticle++;
				}

				/****根据粒子的权值降序排列****/
				pParticle = particles;
				qsort(pParticle, PARTICLE_NUMBER, sizeof(PARTICLE), &particle_decrease);

				/****根据粒子权重重采样粒子****/
				resample(pParticle);
			}//end else
			pParticle = particles;

			/****计算最大权重目标的期望位置，作为跟踪结果****/
			RotatedRect Max_weight_rot = pParticle->rot_rect;

			/****显示跟踪结果****/
			ellipse(frame, Max_weight_rot, Scalar(0, 0, 255));

			after_select_frames++;//总循环每循环一次，计数加1
			if (after_select_frames > 2)//防止跟踪太长，after_select_frames计数溢出
				after_select_frames = 2;
		}

		////显示视频图片到窗口
		imshow("camera2", frame);

		//键盘响应
		c = (char)waitKey(1);
		if (27 == c)//ESC键
			return -1;
	}
	waitKey();
	return 0;
}