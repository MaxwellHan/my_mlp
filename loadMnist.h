unsigned char * loadMnistData(const char * filename,  int &row,  int &col,  int &num);
unsigned char * loadMnistLabel(const char * filename, int &num);
void freeMnist(unsigned char * pData, unsigned char * pLabel);