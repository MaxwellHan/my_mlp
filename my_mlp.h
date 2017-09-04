#pragma once
#include"my_Layers.h"
//网络类，定义网络
class my_mlp{		
public:
	//成员
	unsigned int num_layers;	//层数
	std::vector<my_Layer*>pLayers;	//层的指针
public:
	my_mlp();
	virtual ~my_mlp();
	virtual void buildNet(int m);	//传入输入层大小
	virtual void forward();		//整体前向输出
	virtual void backward();		//后向传播
	virtual void zeroDelta();	//清理梯度
};

