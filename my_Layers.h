#pragma once
#include<vector>
#include <limits>
#include <string>
const double MIN_DOUBLE = std::numeric_limits<double>::min();
const double eps = 1e-12;	//定义一个极小量防止除以0
class my_Layer{
public:
	//定义成员变量
	int n=0;	//本层输出维度数
	int m=0;	//输入数据的维度数
	my_Layer *pIn = nullptr;	//输入层的指针m个
	std::vector<double> data;	//输出数据
	std::vector<double> delta;
	std::string name = "baseLayer";
public:
	my_Layer(){}
	virtual ~my_Layer(){}
	virtual void forward(){};
	virtual void backward(){};
	virtual void zeroDelta();
};
//全链接层
class my_bpLayer:public my_Layer{
public:
	//定义成员变量
	std::vector<std::vector<double> >W;	//权值 n*(m+1) 权值是要给优化方法修改的	
public:
	//定义成员函数
	my_bpLayer(){}
	my_bpLayer(my_Layer * pIn,int n);	//
	virtual ~my_bpLayer(){}
	virtual void forward();
	virtual void backward();
};
//激活层
class my_activeLayer :public my_Layer{
public:
	my_activeLayer(){}
	my_activeLayer(my_Layer * pIn);
	virtual ~my_activeLayer(){}
	virtual void forward();
	virtual void backward();
};
//softmax层
class my_softmaxLayer :public my_Layer{
public:
	my_softmaxLayer(){}
	my_softmaxLayer(my_Layer * pIn);
	virtual ~my_softmaxLayer(){}
	virtual void forward();
	virtual void backward();
};
//输入层
class my_inputLayer :public my_Layer{
public:
	my_inputLayer(){}
	my_inputLayer(double* pIn,int n);
	my_inputLayer(std::vector<double>& pIn, int n);
	virtual ~my_inputLayer(){}
};
//损失层
class my_lossLayer :public my_Layer{
public:
	std::vector<double> *pT=nullptr; //教师信号
public:
	my_lossLayer(){}
	my_lossLayer(my_Layer * pIn);
	virtual ~my_lossLayer(){}
	bool setTeacher(std::vector<double> *pT);	//设置教师信号
	virtual void forward();
	virtual void backward();
};