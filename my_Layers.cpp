#include "my_Layers.h"
#include <assert.h>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
//基础类 层类
void my_Layer::zeroDelta(){
	int n = delta.size();
	for (int i = 0; i < n;++i){
		delta[i] = 0;
	}
}
//输入层
my_inputLayer::my_inputLayer(double* pIn, int n){
	name = "inputLayer";
	data = std::vector<double>(pIn,pIn+n);
	this->m = n;
	this->n = n;
}
my_inputLayer::my_inputLayer(std::vector<double>& vIn, int n){
	name = "inputLayer";
	data = vIn;
	this->m = n;
	this->n = n;
}
//损失层
my_lossLayer::my_lossLayer(my_Layer * pIn){
	name = "lossLayer";
	this->pIn = pIn;
	this->m = pIn->n;
	this->n = 1;
	data = std::vector<double>(n);
	delta = std::vector<double>(n);
}

bool my_lossLayer::setTeacher(std::vector<double> *vT){
	if (vT->size() != m){
		return false;
	}
	this->pT = vT;
}
void my_lossLayer::forward(){
	assert(m > 1);
	assert(n == 1);
	assert(pIn->data.size() == m);
	assert(pT->size() == m);
	data[0] = 0;
	double tm = 0;
	for (int i = 0; i < m; ++i){
		tm = pIn->data[i] - (*pT)[i];
		data[0] += tm*tm;
	}
	data[0] *= 0.5;
	delta[0] = 1.0;
}
void my_lossLayer::backward(){
	assert(m > 1);
	assert(n == 1);
	assert(pIn->delta.size() == m);
	assert(pT->size() == m);
	for (int i = 0; i < m; ++i){
		pIn->delta[i] = pIn->data[i] - (*pT)[i];
	}
}

//激活层
my_activeLayer::my_activeLayer(my_Layer * pIn){
	name = "activeLayer";
	this->pIn = pIn;
	this->m = pIn->n;
	this->n = this->m;
	data = std::vector<double>(n);
	delta = std::vector<double>(n);
}

void my_activeLayer::forward(){
	assert(m > 1);
	assert(n > 1);
	assert(m == n);
	assert(pIn->data.size() == m);
	assert(data.size() == n);
	for (int i = 0; i < m; ++i){
		data[i] = 1.0 / (1.0 + exp(-pIn->data[i]));
	}
}

void my_activeLayer::backward(){
	assert(m > 1);
	assert(n > 1);
	assert(m == n);
	assert(delta.size() == n);
	assert(pIn->delta.size() == m);
	assert(data.size() == n);
	for (int i = 0; i < m; ++i){
		(pIn->delta)[i] = delta[i]*data[i]*(1 - data[i]);
	}
}

//激活层
my_softmaxLayer::my_softmaxLayer(my_Layer * pIn){
	name = "activeLayer";
	this->pIn = pIn;
	this->m = pIn->n;
	this->n = this->m;
	data = std::vector<double>(n);
	delta = std::vector<double>(n);
}

void my_softmaxLayer::forward(){
	assert(m > 1);
	assert(n > 1);
	assert(m == n);
	assert(pIn->data.size() == m);
	assert(data.size() == n);
	double sum = 0;
	for (int i = 0; i < m; ++i){
		sum += exp(pIn->data[i]);
	}
	for (int i = 0; i < m; ++i){
		data[i] = exp(pIn->data[i])/(sum+eps); 
	}
}

void my_softmaxLayer::backward(){
	assert(m > 1);
	assert(n > 1);
	assert(m == n);
	assert(delta.size() == n);
	assert(pIn->delta.size() == m);
	assert(data.size() == n);
	for (int i = 0; i < m; ++i){
		(pIn->delta)[i] = delta[i] * data[i] * (1 - data[i]);
	}
}
//全链接层
my_bpLayer::my_bpLayer(my_Layer * pIn,int n){
	name = "bpLayer";
	this->pIn = pIn;
	this->m = pIn->n;
	this->n = n;
	data = std::vector<double>(n);
	delta = std::vector<double>(n);
	W = std::vector<std::vector<double> >(n, std::vector<double>(m + 1));
	//权值初始化
	srand(time(0));
	//产生[-0.1到0.1]的随机浮点数,精度是1/2000
	for (int i = 0; i<W.size(); ++i){
		for (int j = 0; j<W[i].size(); ++j){
			W[i][j] = double(rand() % 2000 - 1000) / 10000.0;
		}
	}
}

void my_bpLayer::forward(){
	assert(m > 1);
	assert(n > 1);
	assert(pIn->data.size() == m);
	assert(data.size() == n);
	for (int i = 0; i < n; ++i){
		data[i] = 0;
		for (int j = 0; j < m; ++j){
			data[i] += pIn->data[j] * W[i][j];
		}
		data[i] += W[i][m];	//偏置
	}

}
void my_bpLayer::backward(){
	assert(m > 1);
	assert(n > 1);
	assert(pIn->delta.size() == m);
	assert(delta.size() == n);
	//可以优化
	for (int j = 0; j < m; ++j){
		pIn->delta[j] = 0;	//delta会累加
		for (int i = 0; i < n; ++i){
			pIn->delta[j] += W[i][j] * delta[i];
		}
	}
}