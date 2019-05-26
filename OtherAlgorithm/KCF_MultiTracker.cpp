#include<iostream>
#include<opencv2\opencv.hpp>
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\objdetect.hpp>
#include<opencv2\imgproc.hpp>
#include<opencv2\tracking.hpp>
using namespace cv;
using namespace std;
//Mat frame_c, frame_gray, frame_qualize, frame_2;

//int nThreshold = 30;

//调节亮度
//void on_trackbar(int, void*)
//{
//	/*apos=pos;*/
//	threshold(frame_gray, frame_2, nThreshold, 255, CV_THRESH_BINARY);
//	//imshow("二值化", frame_2);
//	findContours(frame_2, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
//	Mat dstImage = Mat::zeros(frame_gray.size(), CV_8UC3);
//	for (int i = 0; i < hierarchy.size(); i++)
//	{
//		Scalar color = Scalar(rand() % 255, rand() % 255, rand() % 255);
//		drawContours(dstImage, contours, i, color, CV_FILLED, 8, hierarchy);
//	}
//	imshow("二值化", dstImage);
//}
//void on_trackbar1(int, void*){
//}

void basicLinearTransformation(Mat &f) {
	double alpha = 1.0;   //1.0-3.0
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

int main()
{
	VideoCapture videoCap(0);
	Mat frame;
	videoCap >> frame;
	//imshow("0", framePre);
	//basicLinearTransformation(framePre);
	MultiTracker trackers;

	vector<Rect> rois;
	selectROIs("input", frame, rois, false);

	vector<Rect2d> obj;
	vector<Ptr<Tracker>> algorithms;
	for (auto i = 0; i < rois.size(); i++) {
		obj.push_back(rois[i]);
		algorithms.push_back(TrackerMOSSE::create());
	}
	//添加目标
	trackers.add(algorithms, frame, obj);
	while (1)
	{
		videoCap >> frame;
		imshow("now", frame);
		if (frame.empty())break;
		trackers.update(frame, obj);
		for (auto j = 0; j < obj.size(); j++) {
			rectangle(frame, obj[j], Scalar(255, 0, 0), 2, 1);
		}
		/*for (auto j = 0; j < trackers.getObjects().size(); j++) {
			rectangle(frame, trackers.getObjects()[j], Scalar(255, 0, 0), 2, 1);
		}*/
		imshow("output", frame);
		int k = waitKey(100);
		if (k == 27)
		{
			break;
		}
	}
	waitKey(1000);

	return 0;

}



