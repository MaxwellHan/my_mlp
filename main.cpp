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
	//可视化
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
	
	//构建神经网络
	my_mlp mlp;
	mlp.buildNet(imgsize);
	//超参数
	double ita = 0.01;	
	double u = 0.1;
	double v = 0.01;
	int batch_size = 18;
	//构建训练对象
	my_optim opt(&mlp,ita,u,v,batch_size);
	std::vector<double> &pIn = mlp.pLayers[0]->data;


	//////开始训练//////////////
	unsigned char * pImg = pData;
	//取出损失层
	my_lossLayer * ploss = dynamic_cast<my_lossLayer *>(mlp.pLayers[mlp.num_layers - 1]);
	vector<double>& pOut = mlp.pLayers[mlp.num_layers - 2]->data;
	int id_img = 0;
	double loss_batch = 0;
	int id = 0;
	double max = pOut[0];
	//训练的轮数
	int epoch = 1;
	int ie = 0;
	//设置输出流
	cout << setiosflags(ios::fixed) << setprecision(3);
	while (ie<epoch&&id_img < num){
		loss_batch = 0;
		for (int i = 0; i < batch_size; ++i){
			pImg = pData + id_img*imgsize;
			//取一幅图像
			for (int i = 0; i < imgsize; ++i){
				pIn[i] = (double)pImg[i]/255.0;
			}
			//取得图像的label
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
			
			
			//一次传播往返
			ploss->setTeacher(&pT);//要先设置教师信号
			for (auto i : pT){
				cout << i << " ";
			}
			cout << endl;

			mlp.forward();
			
			//预测值
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
			
			cout << ploss->data[0] << " " << (int)pLabel[id_img]<<" "<< id << " " << endl;	//取得损失
			
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

	//释放空间
	delete[] pData;
	delete[] pLabel;

	//***********预测*************************************************
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
		//取一幅图像
		for (int i = 0; i < imgsize; ++i){
			pIn[i] = (double)pImg[i] / 255.0;
		}
		//取得图像的label
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

		//一次传播往返
		ploss->setTeacher(&pT);//要先设置教师信号
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