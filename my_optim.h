#pragma once
#include "my_Layers.h"
#include"my_mlp.h"
struct optim_stc{
	my_bpLayer* pLayer;
	std::vector<std::vector<double> > mt;	//冲量矩阵n*m
	std::vector<std::vector<double> > nt;	//二阶矩阵
	std::vector<std::vector<double> > gt;	//记录梯度的改变的累加
};
class my_optim{
public:
	my_optim() :ita(0.1),u(0.5),v(0.5),batch_size(1.0){};
	my_optim(my_mlp * pNet,double ita,double u,double v,double batch_size);
	virtual ~my_optim();
	virtual void adjustW(); //调整权值
	virtual void update();	//跟新冲量矩阵。
public:
	std::vector<optim_stc*> vOpt;	//所有需要调整权值的层和冲量项等
	double ita;	//学习率
	double u;	//冲量项的权
	double v;	//梯度二阶的权
	double batch_size;
};