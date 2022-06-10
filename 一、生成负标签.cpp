#include<iostream>
#include<fstream>
#include<string>
using namespace std;

int main()
{
	string str;
	char temp;
	int times, n;			//循环移位次数
	//fstream inf("d://test1.txt", ios::in);
	//fstream inf("d://VDJdb+.txt", ios::in);
	//fstream outf("d://VDJdb-.txt", ios::out);
	fstream inf("d://McPAS+.txt", ios::in);
	fstream outf("d://McPAS-.txt", ios::out);    
	if (!inf)
	{
		cout << "+打开失败" << endl;
	}
	if (!outf)
	{
		cout << "-打开失败" << endl;
	}
	
	while (inf >> str)
	{
		n = str.length() - 3;
		times = (rand() % (n - 2)) + 2;		//循环移位次数为随机生成[2，str.length()-3)的整数
		//cout << times << endl;
		for (int j = 0; j < times;j++)		//循环移位times次
		{
			temp = str[2];
			for (int i = 2; i < str.length() - 2; i++)
			{
				str[i] = str[i + 1];
			}
			str[str.length() - 2] = temp;
		}
		outf << str << endl;
	}

	inf.close();
	outf.close();
	system("pause");
	return 0;
}
