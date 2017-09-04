#pragma once
#include"my_Layers.h"
//�����࣬��������
class my_mlp{		
public:
	//��Ա
	unsigned int num_layers;	//����
	std::vector<my_Layer*>pLayers;	//���ָ��
public:
	my_mlp();
	virtual ~my_mlp();
	virtual void buildNet(int m);	//����������С
	virtual void forward();		//����ǰ�����
	virtual void backward();		//���򴫲�
	virtual void zeroDelta();	//�����ݶ�
};

