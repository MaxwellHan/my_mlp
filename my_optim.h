#pragma once
#include "my_Layers.h"
#include"my_mlp.h"
struct optim_stc{
	my_bpLayer* pLayer;
	std::vector<std::vector<double> > mt;	//��������n*m
	std::vector<std::vector<double> > nt;	//���׾���
	std::vector<std::vector<double> > gt;	//��¼�ݶȵĸı���ۼ�
};
class my_optim{
public:
	my_optim() :ita(0.1),u(0.5),v(0.5),batch_size(1.0){};
	my_optim(my_mlp * pNet,double ita,double u,double v,double batch_size);
	virtual ~my_optim();
	virtual void adjustW(); //����Ȩֵ
	virtual void update();	//���³�������
public:
	std::vector<optim_stc*> vOpt;	//������Ҫ����Ȩֵ�Ĳ�ͳ������
	double ita;	//ѧϰ��
	double u;	//�������Ȩ
	double v;	//�ݶȶ��׵�Ȩ
	double batch_size;
};