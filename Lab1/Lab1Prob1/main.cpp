/*
Author: Hena Naaz
Class: ECE6122 (A)
Last Date Modified: 26 Sept
Description: Lab1Prob1
Purpose of this file: main function to implement an application that uses command line arguments to input the number of rows and the number of columns of the grid maze.
Output the total number of possible unique paths to a text file (NumberPaths.txt). 
The first command line argument is the number of rows and the second argument is the number of columns. 
The output file is overwritten each time the program is executed.
*/

#include <iostream>
#include <fstream>
#include <limits>
#include <iomanip>
#include <sys/time.h>

using namespace std;

#include "numberGridPaths.cpp"

/**
 * Function readInput() takes arguments and place them as input rows and columns
 * and checking the validity of rows and columns, so that user will know when the inputs were invalid.
 *
 * @param argc integer total arguments entered by user
 * @param argv an array of character pointers
 * @param rows integer updating number of rows in matrix from user input
 * @param cols integer updating number of columns in matrix from user input
 * @return unsigned integer total possible paths to reach the target 
 */
void readInput(int argc, char* argv[], int64_t& rows, int64_t& cols) 
{
	for (int i = 1; i < argc; i++) 
	{
		for (int j = 0; argv[i][j] != '\0'; j++) 
		{
			// If any character other from 0-9 or first character is 0, then marking rows and columns as -1
			if (argv[i][j] < '0' || argv[i][j]>'9' || argv[i][0] == '0') 
			{
				rows = -1;
				cols = -1;
				return;
			}

			// Logic to convert character array to integer values
			if (i == 1)
				rows = rows * 10 + (argv[i][j] - '0');
			else
				cols = cols * 10 + (argv[i][j] - '0');
		}
	}
}

int main(int argc, char* argv[]) 
{
	int64_t rows = 0;
	int64_t cols = 0;

	//Taking user input to check for valid values for the unique path calculation
	readInput(argc, argv, rows, cols);

	ofstream myOutFile("NumberPaths.txt", ofstream::out);
	
	if (argc < 3 || argc > 3 || rows == -1 || cols == -1) 
	{
		myOutFile << "Invalid Input!\n";
		return 0;
	}
	else
	{
		// writing the number of unique paths from function numberGridPaths() to the output file
		uint64_t result = numberGridPaths(rows, cols);
		myOutFile << "Total Number Paths: " << result << endl;

	}
	myOutFile.close();
	
	return 0;
}

