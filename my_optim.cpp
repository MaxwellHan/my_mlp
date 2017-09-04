#include "my_optim.h"
my_optim::my_optim(my_mlp * pNet,double ita,double u,double v,double batch_size){
	this->ita = ita;
	this->u = u;
	this->v = v;
	this->batch_size = batch_size;
	for (auto i : pNet->pLayers){
		if (i->name == "bpLayer"){
			optim_stc *tm = new optim_stc;
			tm->pLayer = dynamic_cast<my_bpLayer*>(i);
			tm->mt = std::vector<std::vector<double> >(i->n, std::vector<double>(i->m+1));
			tm->nt = std::vector<std::vector<double> >(i->n, std::vector<double>(i->m+1));
			tm->gt = std::vector<std::vector<double> >(i->n, std::vector<double>(i->m + 1));
			vOpt.push_back(tm);
		}
	}
}
my_optim::~my_optim(){
	for (auto Opt : vOpt){
		delete Opt;
	}
}
void my_optim::adjustW(){
	//double delta_t = 0;
	double gt = 0;
	for (auto opt : vOpt){
		for (int i = 0; i < opt->pLayer->n; ++i){
			//delta_t = opt->pLayer->delta[i];
			//std::vector<double> & pData = opt->pLayer->pIn->data;
			int m = opt->pLayer->m;
			for (int j = 0; j <m; ++j){
				//gt = delta_t*pData[j];
				gt = opt->gt[i][j]/batch_size;
				opt->gt[i][j] = 0;//调整完置零
				opt->mt[i][j] = u*opt->mt[i][j]/(1.0-v) + gt;
				opt->nt[i][j] = sqrt(v*opt->nt[i][j]/(1.0 - v)+ gt*gt);
				opt->pLayer->W[i][j] -= ita*opt->mt[i][j] / (eps + opt->nt[i][j]);	
			}
			//gt = delta_t;	//调整偏置的权
			gt = opt->gt[i][m] / batch_size;
			opt->mt[i][m] = u*opt->mt[i][m] / (1.0 - v) + gt;
			opt->nt[i][m] = sqrt(v*opt->nt[i][m] / (1.0 - v) + gt*gt);
			opt->pLayer->W[i][m] -= ita*opt->mt[i][m] / (eps + opt->nt[i][m]);
		}
	}
}
void my_optim::update(){
	for (auto opt : vOpt){
		for (int i = 0; i < opt->pLayer->n; ++i){
			double delta_t = opt->pLayer->delta[i];
			std::vector<double> & pData = opt->pLayer->pIn->data; //下一层的数据
			int m = opt->pLayer->m;
			for (int j = 0; j <m; ++j){
				opt->gt[i][j] += delta_t*pData[j];
				//opt->mt[i][j] = u*opt->mt[i][j] / (1.0 - v) + gt;
				//opt->nt[i][j] = sqrt(v*opt->nt[i][j] / (1.0 - v) + gt*gt);
				//opt->pLayer->W[i][j] -= ita*opt->mt[i][j] / (eps + opt->nt[i][j]);
			}
			opt->gt[i][m] = delta_t;	//调整偏置的权
			//opt->mt[i][m] = u*opt->mt[i][m] / (1.0 - v) + gt;
			//opt->nt[i][m] = sqrt(v*opt->nt[i][m] / (1.0 - v) + gt*gt);
			//opt->pLayer->W[i][m] -= ita*opt->mt[i][m] / (eps + opt->nt[i][m]);
		}
	}
}