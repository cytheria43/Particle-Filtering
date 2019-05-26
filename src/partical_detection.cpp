#include <iostream>
#include<vector>
#include"partical.h"

Rect select;//ѡ������cv::Rect������
vector<RotatedRect> select_rotate;
Point origin;//���ĵ㡣opencv���ṩ�ĵ��ģ����
bool select_flag = false;//ѡ���־
bool tracking = false;//���ٱ�־�������ж��Ƿ��Ѿ�ѡ����Ŀ��λ�ã�ֻ����Ϊtrueʱ���ſ�ʼ����
bool select_show = false;
Mat frame, hsv;//ȫ�ֱ�������
int after_select_frames = 0;
vector<vector<Point>>contours;
vector<Vec4i>hierarchy;

string window_name = "select";

void detectContour(Mat &fr, Mat &result) {

	threshold(fr, fr, 45, 255, CV_THRESH_BINARY);
	//imshow("��ֵ��", frame);
	Mat elem1 = getStructuringElement(MORPH_RECT, Size(2, 2));
	Mat elem2 = getStructuringElement(MORPH_RECT, Size(2, 2));
	erode(fr, fr, elem1);//��ʴ
	dilate(fr, fr, elem2);//����

	findContours(fr, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//Mat dstImage = Mat::zeros(frame.size(), CV_8UC3);
	//for (int i = 0; i < hierarchy.size(); i++)
	//{
	//	Scalar color = Scalar(rand() % 255, rand() % 255, rand() % 255);
	//	drawContours(dstImage, contours, i, color, CV_FILLED, 8, hierarchy);
	//}
	//imshow("����ͼ", dstImage);
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
			line(frame, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //������С��Ӿ���ÿ����
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
	//namedWindow("���ȶԱȶ�");
	//createTrackbar("����", "���ȶԱȶ�", &beta, 100, on_trackbar1);
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
	//��������ʼ���˲�ģ��
	Mat kernel(3, 3, CV_32F, cv::Scalar(0));
	kernel.at<float>(1, 1) = 8.0;
	kernel.at<float>(0, 1) = -1.0;
	kernel.at<float>(1, 0) = -1.0;
	kernel.at<float>(1, 2) = -1.0;
	kernel.at<float>(2, 1) = -1.0;
	//��ͼ������˲�
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
	frame = imread(filename); //��һ֡
	/******ͨ���Ե�һ֡����Ԥ����õ�����ɢ������ת����*********/
	Mat gg = frame.clone();
	sharpen(gg);
	cvtColor(gg, gg, CV_RGB2GRAY);
	bitwise_not(gg, gg);
	detectContour(gg, gg);
	imshow("selectrotate", frame);
	Rect select_copy = select;
	while (1)
	{
		/****��ȡһ֡ͼ��****/
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
		/****��rgb�ռ�ת��Ϊhsv�ռ�****/
		cvtColor(frame, hsv, CV_BGR2HSV);

		if (tracking)
		{

			if (1 == after_select_frames)//ѡ����Ŀ�������
			{
				/****��ʼ��Ŀ������****/
				pParticle = particles;//ָ���ʼ��ָ��particles����
				initializationOFpartical(hsv,select_rotate[0], target_hist, pParticle);
				
			}
			else if (2 == after_select_frames)//�ӵڶ�֡��ʼ�Ϳ��Կ�ʼ������
			{
				double sum = 0.0;
				pParticle = particles;
				RNG rng;//�����������

				/****�������ӽṹ��Ĵ󲿷ֲ���****/
				update_PARTICLES(hsv,pParticle, rng, track_hist, target_hist, sum);

				/****��һ������Ȩ��****/
				pParticle = particles;
				for (int i = 0; i < PARTICLE_NUMBER; i++)
				{
					pParticle->weight /= sum;
					pParticle++;
				}

				/****�������ӵ�Ȩֵ��������****/
				pParticle = particles;
				qsort(pParticle, PARTICLE_NUMBER, sizeof(PARTICLE), &particle_decrease);

				/****��������Ȩ���ز�������****/
				resample(pParticle);
			}//end else
			pParticle = particles;

			/****�������Ȩ��Ŀ�������λ�ã���Ϊ���ٽ��****/
			RotatedRect Max_weight_rot = pParticle->rot_rect;

			/****��ʾ���ٽ��****/
			ellipse(frame, Max_weight_rot, Scalar(0, 0, 255));

			after_select_frames++;//��ѭ��ÿѭ��һ�Σ�������1
			if (after_select_frames > 2)//��ֹ����̫����after_select_frames�������
				after_select_frames = 2;
		}

		////��ʾ��ƵͼƬ������
		imshow("camera2", frame);

		//������Ӧ
		c = (char)waitKey(1);
		if (27 == c)//ESC��
			return -1;
	}
	waitKey();
	return 0;
}