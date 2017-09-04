#pragma once
#include<vector>
#include <limits>
#include <string>
const double MIN_DOUBLE = std::numeric_limits<double>::min();
const double eps = 1e-12;	//����һ����С����ֹ����0
class my_Layer{
public:
	//�����Ա����
	int n=0;	//�������ά����
	int m=0;	//�������ݵ�ά����
	my_Layer *pIn = nullptr;	//������ָ��m��
	std::vector<double> data;	//�������
	std::vector<double> delta;
	std::string name = "baseLayer";
public:
	my_Layer(){}
	virtual ~my_Layer(){}
	virtual void forward(){};
	virtual void backward(){};
	virtual void zeroDelta();
};
//ȫ���Ӳ�
class my_bpLayer:public my_Layer{
public:
	//�����Ա����
	std::vector<std::vector<double> >W;	//Ȩֵ n*(m+1) Ȩֵ��Ҫ���Ż������޸ĵ�	
public:
	//�����Ա����
	my_bpLayer(){}
	my_bpLayer(my_Layer * pIn,int n);	//
	virtual ~my_bpLayer(){}
	virtual void forward();
	virtual void backward();
};
//�����
class my_activeLayer :public my_Layer{
public:
	my_activeLayer(){}
	my_activeLayer(my_Layer * pIn);
	virtual ~my_activeLayer(){}
	virtual void forward();
	virtual void backward();
};
//softmax��
class my_softmaxLayer :public my_Layer{
public:
	my_softmaxLayer(){}
	my_softmaxLayer(my_Layer * pIn);
	virtual ~my_softmaxLayer(){}
	virtual void forward();
	virtual void backward();
};
//�����
class my_inputLayer :public my_Layer{
public:
	my_inputLayer(){}
	my_inputLayer(double* pIn,int n);
	my_inputLayer(std::vector<double>& pIn, int n);
	virtual ~my_inputLayer(){}
};
//��ʧ��
class my_lossLayer :public my_Layer{
public:
	std::vector<double> *pT=nullptr; //��ʦ�ź�
public:
	my_lossLayer(){}
	my_lossLayer(my_Layer * pIn);
	virtual ~my_lossLayer(){}
	bool setTeacher(std::vector<double> *pT);	//���ý�ʦ�ź�
	virtual void forward();
	virtual void backward();
};