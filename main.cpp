#include<iostream>
#include"my_mlp.h"
#include"loadMnist.h"
#include"opencv.hpp"
#include"my_optim.h"
#include <iomanip>
using namespace std;

int main(){
	//
	int num,numl, row, col;
	unsigned char *pData = loadMnistData("E:\\DataSets\\Mnist\\train-images.idx3-ubyte", row,col,num);
	unsigned char *pLabel = loadMnistLabel("E:\\DataSets\\Mnist\\train-labels.idx1-ubyte", numl);
	int imgsize = row*col;
	if (numl != num){
		return 0;
	}
	cout << num << " " << row << " " << col << endl;
	//���ӻ�
	//int R, C;
	//if (num == 60000){
	//	R = 300;
	//	C = 200;
	//}
	//else if (num == 10000){
	//	R = 100; C = 100;
	//}
	//cv::Mat img(R*row, C*col, CV_8UC1);

	//for (int r = 0; r < R; ++r){
	//	for (int c = 0; c < C; ++c){
	//		//cout <<(int) pLabel[r*C + c] << " ";
	//		for (int i = 0; i < row; ++i){
	//			for (int j = 0; j < col; ++j){
	//				img.at<unsigned char>(r*row+i,c*col+j) = *(pData +(r*C+c)*imgsize+ i*col + j);
	//				//cout << (int)img.at<unsigned char>(i, j) << " ";
	//			}
	//			//cout << endl;
	//		}
	//	}
	//	//cout << endl;
	//}
	//cv::namedWindow("img",0);
	//cv::imshow("img", img);
	//cv::waitKey();
	
	//����������
	my_mlp mlp;
	mlp.buildNet(imgsize);
	//������
	double ita = 0.01;	
	double u = 0.1;
	double v = 0.01;
	int batch_size = 18;
	//����ѵ������
	my_optim opt(&mlp,ita,u,v,batch_size);
	std::vector<double> &pIn = mlp.pLayers[0]->data;


	//////��ʼѵ��//////////////
	unsigned char * pImg = pData;
	//ȡ����ʧ��
	my_lossLayer * ploss = dynamic_cast<my_lossLayer *>(mlp.pLayers[mlp.num_layers - 1]);
	vector<double>& pOut = mlp.pLayers[mlp.num_layers - 2]->data;
	int id_img = 0;
	double loss_batch = 0;
	int id = 0;
	double max = pOut[0];
	//ѵ��������
	int epoch = 1;
	int ie = 0;
	//���������
	cout << setiosflags(ios::fixed) << setprecision(3);
	while (ie<epoch&&id_img < num){
		loss_batch = 0;
		for (int i = 0; i < batch_size; ++i){
			pImg = pData + id_img*imgsize;
			//ȡһ��ͼ��
			for (int i = 0; i < imgsize; ++i){
				pIn[i] = (double)pImg[i]/255.0;
			}
			//ȡ��ͼ���label
			std::vector<double> pT(10);
			pT[pLabel[id_img]] = 1.0;

			//**************
			cv::Mat img(row, col, CV_8UC1);
			for (int i = 0; i < row; ++i){
				for (int j = 0; j < col; ++j){
					img.at<unsigned char>(i, j) = *(pImg + i*col+j);

				}
			}
			cv::imshow("img", img);
			cv::waitKey(10);
			//****************
			
			
			//һ�δ�������
			ploss->setTeacher(&pT);//Ҫ�����ý�ʦ�ź�
			for (auto i : pT){
				cout << i << " ";
			}
			cout << endl;

			mlp.forward();
			
			//Ԥ��ֵ
			max = pOut[0];
			id = 0;
			cout << pOut[0] << " ";
			for (int i = 1; i<10; ++i){
				cout << pOut[i] << " ";
				if (pOut[i] > max){
					max = pOut[i];
					id = i;
				}
			}
			cout << endl;
			
			cout << ploss->data[0] << " " << (int)pLabel[id_img]<<" "<< id << " " << endl;	//ȡ����ʧ
			
			mlp.backward();
			opt.update();
			loss_batch += ploss->data[0];
			++id_img;
		}
		
		cout << loss_batch / (double)batch_size  << " " << id << " " << (int)pLabel[id_img] << " " << id_img << "/" << num << endl;

		//id_img = 0;
		opt.adjustW();
		//mlp.zeroDelta();
		if (id_img+batch_size >= num){
			id_img = id_img+batch_size-num;
			++ie;
			continue;
		}
	}

	//�ͷſռ�
	delete[] pData;
	delete[] pLabel;

	//***********Ԥ��*************************************************
	pData = loadMnistData("E:\\DataSets\\Mnist\\t10k-images.idx3-ubyte", row, col, num);
	pLabel = loadMnistLabel("E:\\DataSets\\Mnist\\t10k-labels.idx1-ubyte", numl);
	imgsize = row*col;
	if (numl != num){
		return 0;
	}
	cout << num << " " << row << " " << col << endl;
	id_img = 0;
	int numc = 0;
	while (id_img < num){
		loss_batch = 0;
		pImg = pData + id_img*imgsize;
		//ȡһ��ͼ��
		for (int i = 0; i < imgsize; ++i){
			pIn[i] = (double)pImg[i] / 255.0;
		}
		//ȡ��ͼ���label
		std::vector<double> pT(10);
		pT[pLabel[id_img]] = 1.0;

		//**************
		cv::Mat img(row, col, CV_8UC1);
		for (int i = 0; i < row; ++i){
			for (int j = 0; j < col; ++j){
				img.at<unsigned char>(i, j) = *(pImg + i*col + j);

			}
		}
		cout << (int)pLabel[id_img] << endl;
		cv::imshow("img", img);
		cv::waitKey(10);
		//****************

		//һ�δ�������
		ploss->setTeacher(&pT);//Ҫ�����ý�ʦ�ź�
		mlp.forward();

		max = pOut[0];
		id = 0;
		for (int i = 1; i<10; ++i){
			if (pOut[i] > max){
				max = pOut[i];
				id = i;
			}
		}

		cout << (int)pLabel[id_img] << " "<<id << " " ;
		if (id == pLabel[id_img]){
			++numc;
		}
		cout << double(numc) / double(id_img + 1) <<" "<< id_img << "/" << num << endl;

		++id_img;
	}


	cv::waitKey();
	freeMnist(pData, pLabel);
	return 0;
}