#pragma once
#pragma execution_character_set("utf-8")
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include<math.h>
#include<vector>
using namespace cv;
using namespace std;
/****定义使用粒子数目宏****/
#define PARTICLE_NUMBER 100 //
typedef struct particle
{
	int orix, oriy;//原始粒子坐标
	int x, y;//当前粒子的坐标
	double scale;//当前粒子窗口的尺寸
	int prex, prey;//上一帧粒子的坐标
	double prescale;//上一帧粒子窗口的尺寸
	RotatedRect rot_rect;
	Mat hist;//当前粒子窗口直方图特征
	double weight;//当前粒子权值
}PARTICLE;
//****有关粒子窗口变化用到的相关变量 Rob Hess****/
int A1 = 2;
int A2 = -1;
int B0 = 1;
double sigmax = 1.0;
double sigmay = 0.5;
double sigmas = 0.001;

//*************************************hsv空间用到的变量***************************//
int channels[] = { 0,1,2 };//需要统计的通道dim，第一个数组通道从0到image[0].channels()-1，第二个数组从image[0].channels()到images[0].channels()+images[1].channels()-1，
int ZhiFangTuWeiShu = 3;//需要计算直方图的维度
int hist_size[] = { 16,16,16 };// 每个维度的直方图尺寸的数组 
float hrange[] = { 0,180.0 };//色调H的取值范围
float srange[] = { 0,256.0 };//饱和度S的取值范围
float vrange[] = { 0,256.0 };//亮度V的取值范围
const float *ranges[] = { hrange,srange,vrange };//每个维度中bin的取值范围 
float range_angle[] = { -5,0,5 };

extern PARTICLE particles[PARTICLE_NUMBER];


/******判断点是否在旋转矩阵内*******/
bool find_the_y(RotatedRect rot, Point2f p) {
	vector<Point> contour;
	Point2f con[4];
	rot.points(con);
	for (int i = 0; i < 4; i++) {
		contour.push_back(con[i]);
	}
	bool inner=false;
	pointPolygonTest(contour, p, inner);
	return inner;

}
/******根据旋转矩阵截取感兴趣区域并计算直方图*******/
void cal_hist(Mat hsv, RotatedRect rotate_rect, Mat &hist) {
	Mat rot_dst, img;
	Mat rot = getRotationMatrix2D(rotate_rect.center, rotate_rect.angle, 1);
	warpAffine(hsv, rot_dst, rot, hsv.size(), CV_INTER_CUBIC);//旋转
	imshow("rotated", hsv);
	getRectSubPix(rot_dst, rotate_rect.size, rotate_rect.center, img);
	imshow("矫正后", img);
	calcHist(&img, 1, channels, Mat(), hist, 3, hist_size, ranges);
	normalize(hist, hist);

}
/***初始化粒子****/
void initializationOFpartical(Mat hsv, RotatedRect select_rotate,Mat &target_hist,PARTICLE *pParticle) {
	//利用rotaterect做初始化
	RotatedRect select = select_rotate;
	cal_hist(hsv, select, target_hist);

	/****初始化目标粒子****/
	pParticle = particles;//指针初始化指向particles数组


	for (int x = 0; x < PARTICLE_NUMBER; x++)
	{
		pParticle->x = cvRound(select.center.x);//选定目标矩形框中心为初始粒子窗口中心
		pParticle->y = cvRound(select.center.y);
		pParticle->orix = pParticle->x;//粒子的原始坐标为选定矩形框(即目标)的中心
		pParticle->oriy = pParticle->y;
		pParticle->prex = pParticle->x;//更新上一次的粒子位置
		pParticle->prey = pParticle->y;

		pParticle->rot_rect = select;
		//自定义
		pParticle->prescale = 1;
		pParticle->scale = 1;
		pParticle->hist = target_hist;
		pParticle->weight = 0;
		pParticle++;
	}
}
/*****更新粒子位置*****/
void update_PARTICLES(Mat hsv,PARTICLE* pParticle, RNG& rng, Mat& track_hist, Mat& target_hist, double& sum)
{
	int xpre, ypre;
	double pres, s;
	int x, y;
	for (int lizishu = 0; lizishu < PARTICLE_NUMBER; lizishu++)
	{
		//当前粒子的坐标
		xpre = pParticle->x;
		ypre = pParticle->y;

		//当前粒子窗口的尺寸
		pres = pParticle->scale;

		/*更新跟踪rotated矩形框中心，即粒子中心*///使用二阶动态回归来自动更新粒子状态
		RotatedRect rec = pParticle->rot_rect;
		Rect bund_rect = pParticle->rot_rect.boundingRect();
		while (1) {
			x = cvRound(A1*(pParticle->x - pParticle->orix) + A2 * (pParticle->prex - pParticle->orix) +
				B0 * rng.gaussian(sigmax) + pParticle->orix);
			
			if (x >= bund_rect.x && x <= bund_rect.x+bund_rect.width) {
				while (1) {
					y = cvRound(A1*(pParticle->y - pParticle->oriy) + A2 * (pParticle->prey - pParticle->oriy) +
						B0 * rng.gaussian(sigmay) + pParticle->oriy);
					if(find_the_y(rec,Point2f(x,y)))break;
				}
				
			}
		}
		pParticle->x = max(0, min(x, hsv.cols - 1));
		pParticle->y = max(0, min(y, hsv.rows - 1));

		s = A1 * (pParticle->scale - 1) + A2 * (pParticle->prescale - 1) + B0 * (rng.gaussian(sigmas)) + 1.0;
		pParticle->scale = max(1.0, min(s, 3.0));//

		pParticle->prex = xpre;
		pParticle->prey = ypre;
		pParticle->prescale = pres;

		/*计算更新得到rotated矩形框数据*/
		//pParticle->rect.x = max(0, min(cvRound(pParticle->x - 0.5*pParticle->scale*pParticle->rect.width), hsv.cols));
		//pParticle->rect.y = max(0, min(cvRound(pParticle->y - 0.5*pParticle->scale*pParticle->rect.height), hsv.rows));
		//pParticle->rect.width = min(cvRound(pParticle->rect.width), hsv.cols - pParticle->rect.x);
		//pParticle->rect.height = min(cvRound(pParticle->rect.height), hsv.rows - pParticle->rect.y);
		pParticle->rot_rect.center = Point(pParticle->x, pParticle->y);
		pParticle->rot_rect.size = rec.size;
		
		for (float i = -5; i < 10; i += 5) {
			RotatedRect r = rec;
			r.angle += i;
			cal_hist(hsv, r, track_hist);
			/*用巴氏系数计算相似度，一直与最初的目标区域相比*/
			if (1.0 - compareHist(target_hist, track_hist, CV_COMP_BHATTACHARYYA) > pParticle->weight) {
				pParticle->rot_rect.angle += i;
				pParticle->weight = 1.0 - compareHist(target_hist, track_hist, CV_COMP_BHATTACHARYYA);
			}
		}

		/*粒子权重累加*/
		sum += pParticle->weight;
		pParticle++;//指针移向下一位
	}
}
/****重采样*******/
void resample(PARTICLE* pParticle) {
	PARTICLE newParticle[PARTICLE_NUMBER];
	int np = 0, k = 0;
	for (int i = 0; i < PARTICLE_NUMBER; i++)
	{
		np = cvRound(pParticle->weight*PARTICLE_NUMBER);
		for (int j = 0; j < np; j++)
		{
			newParticle[k++] = particles[i];
			if (k == PARTICLE_NUMBER)
				goto EXITOUT;
		}
	}
	while (k < PARTICLE_NUMBER)
		newParticle[k++] = particles[0];
EXITOUT:
	for (int i = 0; i < PARTICLE_NUMBER; i++)
		particles[i] = newParticle[i];
}
/****粒子权值降序排列函数****/
int particle_decrease(const void *p1, const void *p2)
{
	PARTICLE* _p1 = (PARTICLE*)p1;
	PARTICLE* _p2 = (PARTICLE*)p2;
	if (_p1->weight < _p2->weight)
		return 1;
	else if (_p1->weight > _p2->weight)
		return -1;
	return 0;//相等的情况下返回0
}



