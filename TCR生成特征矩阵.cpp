#include<iostream>
#include<fstream>
#include<string>
#include<vector>
using namespace std;

int main()
{
	string str;
	string temp;
	int flag;
	int k=1;												//查看进度
	vector<string>duanlian;								//所有的短链

	fstream inf1("D://TCR//TCR+.txt", ios::in);			//所有维度
	fstream inf2("D://TCR//TCR+.txt", ios::in);			//所有维度
	fstream inf3("D://TCR//TCR-.txt", ios::in);			//所有特征矩阵
	fstream inf4("D://TCR//TCR-.txt", ios::in);			//所有特征矩阵
	fstream outf("D://TCR//matrix.txt", ios::out);
	if ((!inf1) || (!inf2))
	{
		cout << "TCR+打开失败" << endl;
	}
	if ((!inf3) || (!inf4))
	{
		cout << "TCR-打开失败" << endl;
	}
	if (!outf)
	{
		cout << "matrix打开失败" << endl;
	}
	//outf << "TCR" << ",";
	while (inf1 >> str)									//输出TCR+所有维度xi
	{
		for (int i = 0; i < str.length() - 2; i++)
		{
			temp = str.substr(i, 3);
			flag = 0;			//flag=0表示str没在duanlian中
			for (int i = 0; i < duanlian.size(); i++)		//查找str是否已经在duanlian中，若在flag=1
			{
				if (temp == duanlian[i])
				{
					flag = 1;
					break;
				}
			}
			if (!flag)				//如果flag=0即str不在duanlian中，插入
			{
				duanlian.push_back(temp);
				//outf << temp << ",";
			}
		}
	}
	cout << "TCR+维度输出完" << endl;
	while (inf3 >> str)									//输出TCR-所有维度xi
	{
		for (int i = 0; i < str.length() - 2; i++)
		{
			temp = str.substr(i, 3);
			flag = 0;										//flag=0表示str没在duanlian中
			for (int i = 0; i < duanlian.size(); i++)		//查找str是否已经在duanlian中，若在flag=1
			{
				if (temp == duanlian[i])
				{
					flag = 1;
					break;
				}
			}
			if (!flag)				//如果flag=0即str不在duanlian中，插入
			{
				duanlian.push_back(temp);
				//outf << temp << ",";
			}
		}
	}
	//outf << "class" << endl;
	cout << "TCR-维度输出完" << endl;
	
	while (inf2 >> str)									//输出每个TCR+的特征矩阵
	{
		//outf << str << " ";
		for (int i = 0; i < duanlian.size(); i++)		//查找str是否已经在duanlian中，若在flag=1
		{
			flag = 0;
			for (int j = 0; j < str.length() - 2; j++)
			{
				temp = str.substr(j, 3);
				if (temp == duanlian[i])
				{
					flag = 1;
					outf << "1" << " ";
					break;
				}
			}
			if (!flag)
				outf << "0" << " ";
		}
		outf << endl;
		//outf << "1" << endl;
		cout << k++ << endl;							//查看进度
	}
	cout << "TCR+特征矩阵输出完" << endl;
	while (inf4 >> str)									//输出每个TCR-的特征矩阵
	{
		//outf << str << " ";
		for (int i = 0; i < duanlian.size(); i++)		//查找str是否已经在duanlian中，若在flag=1
		{
			flag = 0;
			for (int j = 0; j < str.length() - 2; j++)
			{
				temp = str.substr(j, 3);
				if (temp == duanlian[i])
				{
					flag = 1;
					outf << "1" << " ";
					break;
				}
			}
			if (!flag)
				outf << "0" << " ";
		}
		outf << endl;
		//outf << "0" << endl;
		cout << k++ << endl;
	}
	cout << "TCR-特征矩阵输出完" << endl;
	
	inf1.close();
	inf2.close();
	inf3.close();
	inf4.close();
	outf.close();
	system("pause");
	return 0;
}