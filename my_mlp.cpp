#include "my_Layers.h"
#include"my_mlp.h"
#include <assert.h>
my_mlp::my_mlp(){
	num_layers = 0;
}
my_mlp::~my_mlp(){
	for (auto i : pLayers){
		delete i;
	}
}
void my_mlp::forward(){
	for (int i = 1; i < num_layers; ++i){
			pLayers[i]->forward();
	}
}

void my_mlp::backward(){
	for (int i = num_layers - 1; i>1; --i){	//输出层的上一层不往输出层反向传播
			pLayers[i]->backward();
	}
}
void my_mlp::zeroDelta(){
	for (int i = num_layers - 1; i >= 0; --i){
		pLayers[i]->zeroDelta();
	}
}
void my_mlp::buildNet(int m){
	std::vector<double> pIn(m);
	//双隐层
	my_Layer * input = new my_inputLayer(pIn, m);
	pLayers.push_back(input); ++num_layers;
	my_Layer * connect1 = new my_bpLayer(input, 512);
	pLayers.push_back(connect1); ++num_layers;
	my_Layer * active1 = new my_activeLayer(connect1);
	pLayers.push_back(active1); ++num_layers;
	my_Layer * connect2 = new my_bpLayer(active1, 256);
	pLayers.push_back(connect2); ++num_layers;
	my_Layer * active2 = new my_activeLayer(connect2);
	pLayers.push_back(active2); ++num_layers;
	my_Layer * connect3 = new my_bpLayer(active2, 10);
	pLayers.push_back(connect3); ++num_layers;
	//my_Layer * active3 = new my_activeLayer(connect3);
	//pLayers.push_back(active3); ++num_layers;
	my_Layer * softmax = new my_softmaxLayer(connect3);
	pLayers.push_back(softmax); ++num_layers;
	my_Layer * loss = new my_lossLayer(softmax);
	pLayers.push_back(loss); ++num_layers;


	//单隐层
	//my_Layer * input = new my_inputLayer(pIn, m);
	//pLayers.push_back(input); ++num_layers;

	//my_Layer * connect1 = new my_bpLayer(input, 512);
	//pLayers.push_back(connect1); ++num_layers;

	//my_Layer * active1 = new my_activeLayer(connect1);
	//pLayers.push_back(active1); ++num_layers;

	//my_Layer * connect3 = new my_bpLayer(active1, 10);
	//pLayers.push_back(connect3); ++num_layers;

	////my_Layer * active3 = new my_activeLayer(connect3);
	////pLayers.push_back(active3); ++num_layers;
	//my_Layer * softmax = new my_softmaxLayer(connect3);
	//pLayers.push_back(softmax); ++num_layers;

	//my_Layer * loss = new my_lossLayer(softmax);
	//pLayers.push_back(loss); ++num_layers;
}