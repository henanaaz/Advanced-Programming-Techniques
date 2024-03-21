/*
Author: Hena Naaz
Class: ECE6122 (A)
Last Date Modified: 26 Sept
Description: Lab1Prob2
Purpose of this file: implement a program that takes as a command line argument that is the path to an input file. 
There is a sample input file called input_sudoku.txt included in the folder.
*/

#include <iostream>
#include <string>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>
#include <set>
#include <sys/time.h>
#include <iomanip>
using namespace std;

std::fstream outFile;
std::fstream inFile;

mutex outFileMutex;
mutex inFileMutex;
mutex gridSolverCountMutex;
set<int> gridSolved;

/**
 * Class SudokuGrid is used to hold a single puzzle with a constant 9 x 9 array of unsigned char elements.
 * Member variables for SudokuGrid:
 *		 std::string m_strGridName
 *		 unsigned char gridElement[9][9]
 * Member functions for SudokuGrid:
 *		 friend fstream& operator>>(fstream& os, SudokuGrid & gridIn)
 *		 friend fstream& operator<<(fstream& os, const SudokuGrid & gridOut);
 *		 solve();
 * Additional Member functions:
 *		void solve()
 *		bool isValid()
 *		bool helper()
 *		bool isGridSolved
 */
class SudokuGrid {

public:
	string m_strGridName;
	unsigned char gridElement[9][9];

	/*
	* Method Function for SudokuGrid: friend function fstrem &operator>>
	*  	 reads a single SudokuGrid object from a fstream file.
	* 
	* @param os fstream type argument for file
	* @param parent class SudokuGrid type gridIn to operate on objects during input file read
	*/
	friend fstream &operator>>(fstream& os, SudokuGrid& gridIn) 
	{
		
		for(string line; getline(os, line); ) 
		{
			//Checking for every new valid grid entry in the sudoku
			if(line[0] == 'G' && !gridIn.isGridSolved(line)) 
			{
				gridIn.m_strGridName = line;

				// read all 9 rows
				for(int i=0; i<9; i++) 
				{
					line = "";
					getline(os, line);
					
					for(int j=0; j<9; j++) 
					{
						// Updating the object with grid values from input
						gridIn.gridElement[i][j] = line[j];
					}
				}

				return os;
			}
		}

		return os;
	}

	/*
	* Method Function for SudokuGrid: friend function fstrem &operator<<
	*  	  writes the SudokuGrid object to a file in the same format that is used in reading in the object
	*	
	* @param os fstream type argument for file
	* @param parent class SudokuGrid type gridOut to operate on objects during output file write
	*/
	friend fstream &operator<<(fstream& os, const SudokuGrid &gridOut) 
	{

		os<<gridOut.m_strGridName<<endl;

		for (int i = 0; i < 9; i++)
		{
			string str = "";
			for (int j = 0; j < 9; j++)
			{
				// Updating the grid values to gridOut objects for output file
				str.push_back(gridOut.gridElement[i][j]);
			}
			os << str << endl;
		}

		return os;
	}

	// Function solve() is used as a call function for the helper functions that is taking the gridElement entries, starting from 0th row and column
	void solve() 
	{
		helper(gridElement, 0, 0);
	}

	// Function isValid() is used to check if the row and column in each iteration is valid entry
	bool isValid(unsigned char board[9][9], int row, int col, char c) 
	{
	    // grid row check
	    for(int i = 0; i < 9; i++) 
			if(board[i][col] == c) 
				return false;
		// grid column check
	    for(int i = 0; i < 9; i++) 
			if(board[row][i] == c) 
				return false;
	    
		// box check
	    int x0 = (row/3) * 3, y0 = (col/3) * 3;
	    for(int i = 0; i < 3; i++) 
		{
	        for(int j = 0; j < 3; j++) 
			{
	            if(board[x0 + i][y0 + j] == c) return false;
	        }
	    }
	    return true;
	}

	// Function helper() is used to update the gridElements on board, starting from nth rows and columns
	bool helper(unsigned char board[9][9], int row, int col) 
	{
	    // When all rows are done, return the function
	    if(row == 9) 
			return true;

	    // Keep updating for next row
	    if(col == 9) 
			return helper(board, row + 1, 0);

	    // Value in grid already marked
	    if(board[row][col] != '0') 
			return helper(board, row, col + 1);

	    for(char c = '1'; c <= '9'; c++) 
		{
	        if(isValid(board, row, col, c)) 
			{
	            board[row][col] = c;
	            // without return here, the board reverts to initial state
	            if(helper(board, row, col + 1))
					return true;
	            board[row][col] = '0';
	        }
	    }
	    return false;
	}

	// Function isGridSolved() is used
	bool isGridSolved(string str) 
	{
		bool flag = false;
		int gridNumber = 0;
		bool isSolved = false;

		for(int i=0; i<str.size(); i++) 
		{
			if(flag) 
			{
				gridNumber = gridNumber*10 + (str[i]-'0');
			}

			if(str[i] == ' ') 
			{
				flag = true;
			}
		}

		gridSolverCountMutex.lock();

		// solved
		if(gridSolved.find(gridNumber) != gridSolved.end()) 
		{
			isSolved = true;
		}
		else
		{
			gridSolved.insert(gridNumber);
			isSolved = false;
		}

		gridSolverCountMutex.unlock();

		return isSolved;
	}
};

/**
 * Function solveSudokuPuzzles to solve the sudokuGrid and append the output to outFile
 * The function has input file, sudoku solver and output file sections.
 * @return void 
 */
void solveSudokuPuzzles() 
{
	SudokuGrid sudoku;

	do {
		
		//inFile Section:
		//lock inFile mutex
		inFileMutex.lock();
		//check <EOF> for inFile 
		if (inFile.eof())
		{
			break;
		}
		else
		{
			inFile >> sudoku;
		}
		//unlock infile mutex
		inFileMutex.unlock();

		// Sudoku Solver Section:
		// Untill the inFile is not <EOF>, it should solve the grids that is then updated on the outFile
		sudoku.solve();

		//outFile Section:
		//lock outFile mutex
		outFileMutex.lock();
		//update outFile with solved values 
		outFile <<sudoku;
		//unlock out
		outFileMutex.unlock();
	} while (true);

	inFileMutex.unlock();
}


int main(int argc, char *argv[] ) 
{
	// string to handle the input file from user argument
	string filename = argv[1];
	
	//Dynamically determine the number of threads
	unsigned int nthreads = thread::hardware_concurrency();
	vector<thread> threadArray;
	
	// Handling the input file from user input
	inFile.open(filename);

	// Handling the output file Lab2Prob2.txt
	outFile.open("./Lab2Prob2.txt", ios_base::out);

	// Checking if the file exists, to move forward with the Sudoku solver
	if (inFile)
	{
		for (int i = 0; i < nthreads - 1; i++)
		{
			// Taking values from the input file and giving it to the solveSudokuPuzzles() function
			threadArray.push_back(thread(solveSudokuPuzzles));
		}

		for (int i = 0; i < nthreads - 1; i++)
		{
			// Waiting for all threads to complete before returning
			threadArray[i].join();
		}

		//Closing the fstream files before completing the program
		inFile.close();
		outFile.close();
	}
	else
		cout << "The input file does not exist!" << endl;

	return 0;
}
