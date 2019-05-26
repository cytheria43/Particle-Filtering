#pragma once
#pragma execution_character_set("utf-8")
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include<math.h>
#include<vector>
using namespace cv;
using namespace std;
/****����ʹ��������Ŀ��****/
#define PARTICLE_NUMBER 100 //
typedef struct particle
{
	int orix, oriy;//ԭʼ��������
	int x, y;//��ǰ���ӵ�����
	double scale;//��ǰ���Ӵ��ڵĳߴ�
	int prex, prey;//��һ֡���ӵ�����
	double prescale;//��һ֡���Ӵ��ڵĳߴ�
	RotatedRect rot_rect;
	Mat hist;//��ǰ���Ӵ���ֱ��ͼ����
	double weight;//��ǰ����Ȩֵ
}PARTICLE;
//****�й����Ӵ��ڱ仯�õ�����ر��� Rob Hess****/
int A1 = 2;
int A2 = -1;
int B0 = 1;
double sigmax = 1.0;
double sigmay = 0.5;
double sigmas = 0.001;

//*************************************hsv�ռ��õ��ı���***************************//
int channels[] = { 0,1,2 };//��Ҫͳ�Ƶ�ͨ��dim����һ������ͨ����0��image[0].channels()-1���ڶ��������image[0].channels()��images[0].channels()+images[1].channels()-1��
int ZhiFangTuWeiShu = 3;//��Ҫ����ֱ��ͼ��ά��
int hist_size[] = { 16,16,16 };// ÿ��ά�ȵ�ֱ��ͼ�ߴ������ 
float hrange[] = { 0,180.0 };//ɫ��H��ȡֵ��Χ
float srange[] = { 0,256.0 };//���Ͷ�S��ȡֵ��Χ
float vrange[] = { 0,256.0 };//����V��ȡֵ��Χ
const float *ranges[] = { hrange,srange,vrange };//ÿ��ά����bin��ȡֵ��Χ 
float range_angle[] = { -5,0,5 };

extern PARTICLE particles[PARTICLE_NUMBER];


/******�жϵ��Ƿ�����ת������*******/
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
/******������ת�����ȡ����Ȥ���򲢼���ֱ��ͼ*******/
void cal_hist(Mat hsv, RotatedRect rotate_rect, Mat &hist) {
	Mat rot_dst, img;
	Mat rot = getRotationMatrix2D(rotate_rect.center, rotate_rect.angle, 1);
	warpAffine(hsv, rot_dst, rot, hsv.size(), CV_INTER_CUBIC);//��ת
	imshow("rotated", hsv);
	getRectSubPix(rot_dst, rotate_rect.size, rotate_rect.center, img);
	imshow("������", img);
	calcHist(&img, 1, channels, Mat(), hist, 3, hist_size, ranges);
	normalize(hist, hist);

}
/***��ʼ������****/
void initializationOFpartical(Mat hsv, RotatedRect select_rotate,Mat &target_hist,PARTICLE *pParticle) {
	//����rotaterect����ʼ��
	RotatedRect select = select_rotate;
	cal_hist(hsv, select, target_hist);

	/****��ʼ��Ŀ������****/
	pParticle = particles;//ָ���ʼ��ָ��particles����


	for (int x = 0; x < PARTICLE_NUMBER; x++)
	{
		pParticle->x = cvRound(select.center.x);//ѡ��Ŀ����ο�����Ϊ��ʼ���Ӵ�������
		pParticle->y = cvRound(select.center.y);
		pParticle->orix = pParticle->x;//���ӵ�ԭʼ����Ϊѡ�����ο�(��Ŀ��)������
		pParticle->oriy = pParticle->y;
		pParticle->prex = pParticle->x;//������һ�ε�����λ��
		pParticle->prey = pParticle->y;

		pParticle->rot_rect = select;
		//�Զ���
		pParticle->prescale = 1;
		pParticle->scale = 1;
		pParticle->hist = target_hist;
		pParticle->weight = 0;
		pParticle++;
	}
}
/*****��������λ��*****/
void update_PARTICLES(Mat hsv,PARTICLE* pParticle, RNG& rng, Mat& track_hist, Mat& target_hist, double& sum)
{
	int xpre, ypre;
	double pres, s;
	int x, y;
	for (int lizishu = 0; lizishu < PARTICLE_NUMBER; lizishu++)
	{
		//��ǰ���ӵ�����
		xpre = pParticle->x;
		ypre = pParticle->y;

		//��ǰ���Ӵ��ڵĳߴ�
		pres = pParticle->scale;

		/*���¸���rotated���ο����ģ�����������*///ʹ�ö��׶�̬�ع����Զ���������״̬
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

		/*������µõ�rotated���ο�����*/
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
			/*�ð���ϵ���������ƶȣ�һֱ�������Ŀ���������*/
			if (1.0 - compareHist(target_hist, track_hist, CV_COMP_BHATTACHARYYA) > pParticle->weight) {
				pParticle->rot_rect.angle += i;
				pParticle->weight = 1.0 - compareHist(target_hist, track_hist, CV_COMP_BHATTACHARYYA);
			}
		}

		/*����Ȩ���ۼ�*/
		sum += pParticle->weight;
		pParticle++;//ָ��������һλ
	}
}
/****�ز���*******/
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
/****����Ȩֵ�������к���****/
int particle_decrease(const void *p1, const void *p2)
{
	PARTICLE* _p1 = (PARTICLE*)p1;
	PARTICLE* _p2 = (PARTICLE*)p2;
	if (_p1->weight < _p2->weight)
		return 1;
	else if (_p1->weight > _p2->weight)
		return -1;
	return 0;//��ȵ�����·���0
}



