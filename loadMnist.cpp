#include<iostream>
#include<fstream>
#include<algorithm>
using namespace std;
unsigned char * loadMnistData(const char * filename, int &row, int &col, int &num){
	ifstream file(filename,ios::binary);	//二进制形式打开文件
	if (!file)return nullptr;
	unsigned char buffer[4];
	row = 0; col = 0; num = 0;
	file.seekg(0, ios::beg);

	file.read((char*)&buffer,4);
	int type = 0;
	for (int i = 3, k = 1; i >= 0; --i, k *= 256)
		type += k*buffer[i];
	if (type != 2051){
		return nullptr;
	}

	file.read((char*)&buffer, 4);
	for (int i = 3,  k = 1; i >= 0; --i, k *= 256)
		num += k*buffer[i];

	file.read((char*)buffer, 4);
	for (int i = 3,  k = 1; i >= 0; --i, k *= 256)
		row += k*buffer[i];

	file.read((char*)buffer, 4);
	for (int i = 3,  k = 1; i >= 0; --i, k *= 256)
		col += k*buffer[i];

	unsigned char * pData = new unsigned char[num*row*col];
	int size = row*col*num;
	file.read((char*)(pData), size);

	return pData;
}

unsigned char * loadMnistLabel(const char * filename, int &num){
	ifstream file(filename, ios::binary);	//二进制形式打开文件
	if (!file)return nullptr;
	unsigned char buffer[4];
	num = 0;
	file.seekg(0, ios::beg);

	file.read((char*)&buffer, 4);
	int type = 0;
	for (int i = 3, k = 1; i >= 0; --i, k *= 256)
		type += k*buffer[i];
	if (type != 2049){
		return nullptr;
	}

	file.read((char*)&buffer, 4);
	for (int i = 3, k = 1; i >= 0; --i, k *= 256)
		num += k*buffer[i];
	unsigned char * pData = new unsigned char[num];
	file.read((char*)(pData), num); 

	return pData;
}

void freeMnist(unsigned char * pData, unsigned char * pLabel){
	delete[] pData; pData = nullptr;
	delete[] pLabel; pLabel = nullptr;
}