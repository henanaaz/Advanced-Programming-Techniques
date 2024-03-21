#include <iostream>
#include <limits>
#include <vector>
#include <iomanip>
#include <sys/time.h>
using namespace std;

#define ll long long int

void readInput(int argc, char *argv[], int64_t &rows, int64_t &cols) {
	for(int i=1; i<argc; i++) {
		for(int j=0; argv[i][j] != '\0'; j++) {
			if(argv[i][j]<'0' || argv[i][j]>'9' || argv[i][0] == '0') {
				rows =  -1;
				cols =  -1;
				return ;
			}
			if(i==1)
				rows = rows*10+(argv[i][j]-'0');
			else
				cols = cols*10+(argv[i][j]-'0');
		}
	}
}


int main(int argc, char *argv[]) {
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);

	int64_t rows=0;
	int64_t cols=0;

	readInput(argc, argv, rows, cols);

	if(argc < 3 || argc > 3 || rows == -1 || cols == -1) {
		cout<<"Invalid Input\n";
		return 0;
	}

	vector<uint64_t> prevRow(cols, 1);
	uint64_t result;

	if(rows == 1 || cols == 1) {
		result = 1;
	}
	else {
		for(uint64_t i=1; i<rows; i++) {
			for(uint64_t j=1; j<cols; j++) {
				prevRow[j] = prevRow[j]+prevRow[j-1];
			}
		}
		result = prevRow[cols-1];
	}
	

	cout<<result<<endl;
	return 0;
}













