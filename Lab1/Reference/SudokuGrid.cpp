#include <iostream>
#include <string>
#include <fstream>
#include <thread>
#include <vector>
#include <set>
#include <sys/time.h>
#include <iomanip>
using namespace std;

mutex outFileMutex;
mutex inFileMutex;
mutex gridSolverCountMutex;
set<int> gridSolved;

class SudokuGrid {
public:
	string m_strGridName;
	unsigned char gridElement[9][10];


	friend fstream &operator>>(fstream& os, SudokuGrid& gridIn) {
		inFileMutex.lock();
		// cout<<this_thread::get_id()<<" start read:"<<endl;
		for(string line; getline(os, line); ) {
			if(line[0] == 'G' && !gridIn.isGridSolved(line)) {
				gridIn.m_strGridName = line;

				// read all 9 rows
				for(int i=0; i<9; i++) {
					getline(os, line);

					for(int j=0; j<9; j++) {
						gridIn.gridElement[i][j] = line[j];
					}
				}

				// cout<<this_thread::get_id()<<":: "<<temp<<endl;
				// cout<<this_thread::get_id()<<" end read:"<<endl;
				inFileMutex.unlock();
				return os;
			}
		}

		inFileMutex.unlock();
		return os;
	}

	friend fstream &operator<<(fstream& os, const SudokuGrid &gridOut) {
		outFileMutex.lock();
		
		os<<gridOut.m_strGridName<<endl;

		for(int i=0; i<9; i++)
			os<<gridOut.gridElement[i]<<endl;

		outFileMutex.unlock();
		return os;
	}

	void solve() {
		helper(gridElement, 0, 0);
	}

	bool isValid(unsigned char board[9][10], int row, int col, char c) {
	    // row check
	    for(int i = 0; i < 9; i++) 
			if(board[i][col] == c) 
				return false;
		// col check
	    for(int i = 0; i < 9; i++) 
			if(board[row][i] == c) 
				return false;
	    // box check
	    int x0 = (row/3) * 3, y0 = (col/3) * 3;
	    for(int i = 0; i < 3; i++) {
	        for(int j = 0; j < 3; j++) {
	            if(board[x0 + i][y0 + j] == c) return false;
	        }
	    }
	    return true;
	}

	bool helper(unsigned char board[9][10], int row, int col) {
	    // done
	    if(row == 9) 
			return true;
	    // time for next row
	    if(col == 9) 
			return helper(board, row + 1, 0);
	    // already marked
	    if(board[row][col] != '0') 
			return helper(board, row, col + 1);

	    for(char c = '1'; c <= '9'; c++) {
	        if(isValid(board, row, col, c)) {
	            board[row][col] = c;
	            // without return here, the board reverts to initial state
	            if(helper(board, row, col + 1))
					return true;
	            board[row][col] = '0';
	        }
	    }
	    return false;
	}

	bool isGridSolved(string str) {
		bool flag = false;
		int gridNumber = 0;
		bool isSolved = false;

		for(int i=0; i<str.size(); i++) {
			if(flag) {
				gridNumber = gridNumber*10 + (str[i]-'0');
			}

			if(str[i] == ' ') {
				flag = true;
			}
		}

		gridSolverCountMutex.lock();

		// solved
		if(gridSolved.find(gridNumber) != gridSolved.end()) {
			isSolved = true;
		}
		else{
			gridSolved.insert(gridNumber);
			isSolved = false;
		}

		gridSolverCountMutex.unlock();

		return isSolved;
	}
};

void solveSudokuPuzzles() {
	fstream infile;
	infile.open("./input_sudoku.txt");
	SudokuGrid sudokuGrid;

	fstream outfile;
	outfile.open("./output_sudoku.txt", ios_base::app);

	while(infile >> sudokuGrid) {

		// cout<<this_thread::get_id()<<" start solve:"<<endl;
		sudokuGrid.solve();
		// cout<<this_thread::get_id()<<" end solve:"<<endl;


		outfile<<sudokuGrid;
	}
}


int main(int argc, char *argv[] ) {
	unsigned int nthreads = thread::hardware_concurrency();
	vector<thread> threadArray;
	
	for(int i=0; i<nthreads-1; i++) {
		threadArray.push_back( thread(solveSudokuPuzzles) );
	}

	for(int i=0; i<nthreads-1; i++) {
		threadArray[i].join();
	}
	
	return 0;
}
















