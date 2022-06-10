#include<iostream>
#include<algorithm>
#include<cstdlib>
#include<fstream>
#include "Eigen/Dense"
using namespace std;
using namespace Eigen;

void featurenormalize(MatrixXd &X)
{
	//计算每一维度均值
	MatrixXd meanval = X.colwise().mean();
	RowVectorXd meanvecRow = meanval;
	//样本均值化为0
	X.rowwise() -= meanvecRow;
}
void computeCov(MatrixXd &X, MatrixXd &C)
{
	//计算协方差矩阵C = XTX / n-1;
	C = X.adjoint() * X;
	C = C.array() /(X.rows() - 1);
}
void computeEig(MatrixXd &C, MatrixXd &vec, MatrixXd &val)
{
	//计算特征值和特征向量，使用selfadjont按照对阵矩阵的算法去计算，可以让产生的vec和val按照有序排列
	SelfAdjointEigenSolver<MatrixXd> eig(C);

	vec = eig.eigenvectors();
	val = eig.eigenvalues();
}
int computeDim(MatrixXd &val)
{
	int dim;
	double sum = 0;
	for (int i = val.rows() - 1; i >= 0; --i)
	{
		sum += val(i, 0);
		dim = i;

		if (sum / val.sum() >= 0.95)
			break;
	}
	return val.rows() - dim;
}
int main()
{
	ifstream fin("d:\\test1.txt");
	ofstream fout("d:\\test2.txt");
	const int m = 10000, n = 128;
	MatrixXd X(10000, 128), C(128, 128);
	MatrixXd vec, val;

	//读取数据
	double in[200];
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
			fin >> in[j];
		for (int j = 1; j <= n; ++j)
			X(i, j - 1) = in[j - 1];
	}
	//pca

	//零均值化
	featurenormalize(X);
	//计算协方差
	computeCov(X, C);
	//计算特征值和特征向量
	computeEig(C, vec, val);
	//计算损失率，确定降低维数
	int dim = computeDim(val);
	//计算结果
	MatrixXd res = X * vec.rightCols(dim);
	//输出结果
	fout << "the result is " << res.rows() << "x" << res.cols() << " after pca algorithm." << endl;
	fout << res;
	system("pause");
	return 0;
}