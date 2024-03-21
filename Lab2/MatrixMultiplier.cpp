/*
Author: Hena Naaz
Class: ECE6122 (A)
Last Date Modified: 10 October 2022
Description: Lab 2 Problem 1: Matrix Multiplication
What is the purpose of this file?
To write a C++ application that can read in two matrices (A & B) from a text file
using a command line argument to specify the file path and name.
*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <sstream>
#include <iomanip>

#if defined(ENABLE_OPENMP)
#include <omp.h>
#endif

#define ld long double

using namespace std;

//Creating global variables for the input and output matrices
vector<vector<ld>> matrix1, matrix2, matrix3;

/**
 * Function getMatrix(fPointer, rows, cols) is used to take input for all rows and columns and arrange it in matrix vector
 * param @fPointer indicates the file stream containing the values
 * param @rows to take the total number of rows in the matrix
 * param @cols to take the total number of columns in the matrix
**/
vector<vector<ld>> getMatrix(fstream& fPointer, int& rows, int& cols) {
	vector<vector<ld>> matrix;
	string line, word;
	int i = 0;

	//Update rows of the matrix by fetching each line of the input file through fPointer stream
	for (i = 0; i < rows && !fPointer.eof(); i++) 
	{
		getline(fPointer, line);

		istringstream ss(line);
		vector<ld> row;

		int colCount = 0;
		while (ss >> word) 
		{
			int decimalCount = 0;
			for (int k = 0; k < word.size(); k++) 
			{
				if (word[k] == '.') 
				{
					++decimalCount;
				}

				// If the input is non numeric value or non decimal value, then it is invalid
				if (word[k] < '0' && word[k]>'9' && word[k] != '.' || decimalCount > 1) 
				{
					cout << "Invalid matrix input!!!\n";
					exit(0);
				}
			}

			colCount++;
			row.push_back(stold(word));
		}

		//Check to verify the number of columns as present in input file
		if (colCount != cols) 
		{
			cout << "Number of columns not matching!!!\n";
			exit(0);
		}

		matrix.push_back(row);
	}

	//Check to verify the number of rows as present in input file
	if (i != rows) 
	{
		cout << "Number of rows not matching!!!\n";
		exit(0);
	}

	return matrix;
}

/**
 * Function getRowsAndColumns(fPointer, rows, cols) is a function that is used to extract rows and columns, and check for validity
 * param @fPointer takes the fstream for reading the values
 * param @rows represents the number of rows
 * param @cols represents the number of columns
  **/
void getRowsAndColumns(fstream& fPointer, int& rows, int& cols) 
{
	string line;
	getline(fPointer, line);

	for (int i = 0; i < line.size(); i++) 
	{
		if ((line[i] < '0' || line[i] > '9') && line[i] != ' ' && line[i] != 13) 
		{
			cout << "Invalid rows and cols mentioned in file!!!\n";
			exit(0);
		}
	}

	istringstream ss(line);
	string word;

	ss >> word;
	rows = stoi(word);
	ss >> word;
	cols = stoi(word);

	//Check when istringstream is readable
	while (ss >> word) 
	{
		cout << "Invalid Input file!!!\n";
		exit(0);
	}
}

/**
 * Function transpose(matrix) is used to extract a tranposed matrix of the given matrix
 * param @matrix is input vector to take matrix on which transpose will be performed
 **/
vector<vector<ld>> transpose(vector<vector<ld>> matrix) 
{
	vector<vector<ld>> tempMatrix;

	for (int j = 0; j < matrix[0].size(); j++) 
	{
		vector<ld> row;
		for (int i = 0; i < matrix.size(); i++) 
		{
			row.push_back(matrix[i][j]);
		}

		//storing the temporary matrix for transposed matrix
		tempMatrix.push_back(row);
	}

	return tempMatrix;
}

/**
 * Function readFile (filename) is to read the given matrices by using function getRowsAndColumns()
 * param @fileName string is to specify the name of input file stream
 * **/
bool readFile(string fileName) 
{
	fstream fPointer;
	fPointer.open(fileName, ios::in);

	//Chcek if fstream worked properly
	if (fPointer.fail()) 
	{
		return false;
	}

	int rows, cols;

	getRowsAndColumns(fPointer, rows, cols);
	matrix1 = getMatrix(fPointer, rows, cols);

	getRowsAndColumns(fPointer, rows, cols);
	vector<vector<ld>> tempMatrix = getMatrix(fPointer, rows, cols);

	matrix2 = transpose(tempMatrix);

	//As per the size of matrix 1 and matrix 2, the size of matrix 3 is decided
	matrix3.resize(matrix1.size(), vector<ld>(matrix2.size()));

	fPointer.close();

	return true;
}

/**
 * Function matrixMultiply() is used to perform matrix multiplication between gloobal matrix 1 and 2 and calculate in matrix 3
 * **/
void matrixMultiply() 
{
	int i = 0, j = 0, k = 0;

//If the program is runnig in fopenmp mode
#if defined(ENABLE_OPENMP)
	omp_set_num_threads(omp_get_num_procs());
#endif

#pragma omp parallel for private(i,j,k) shared(matrix1,matrix2,matrix3)
	//Loop to read matrix 1 rows
	for (i = 0; i < matrix1.size(); i++) 
	{
		vector<ld> row = matrix1[i];
                //Loop to read matrix 2 columns
		for (j = 0; j < matrix2.size(); j++) 
		{
			vector<ld> col = matrix2[j];

			long double sum = 0.0;

			//Calculate the value of one entry in matrix in the sum, for each rows and columns multiplied
			for (k = 0; k < row.size(); k++) {
				sum += row[k] * col[k];
			}

			matrix3[i][j] = sum;
		}
	}
}

int main(int argc, char* argv[]) 
{
	//Check if the number of arguments are valid
	//Only 1 input file is expected in this program as argument
	if (argc != 2) 
	{
		cout << "Invalid Input!!!\n";
		return 0;
	}

	string fileName(argv[1]);

	bool isFileReadSuccess = readFile(fileName);

	//Check if the input file reading was done sucessfully
	if (isFileReadSuccess) 
	{
		matrixMultiply();
		fstream fPointer;
		//Use MatrixOut.txt as an output file to write the output (muliplied) matrix3 
		fPointer.open("MatrixOut.txt", ios::out);

		if (fPointer.fail()) 
		{
			cout << "Invalid Input!!!\n";
			exit(0);
		}

		//Mention the dimension of the output matrix at the top of the output file
		fPointer << matrix3.size() << " " << matrix3[0].size() << "\n";

		//Loop to extract the rows and columns for the output file
		for (int i = 0; i < matrix3.size(); i++) 
		{
			for (int j = 0; j < matrix3[0].size(); j++) 
			{
				//Set the precision as 7 for the matrix values being written in fPointer output stream
				fPointer << setprecision(7) << fixed << matrix3[i][j] << " ";
			}
			fPointer << "\n";
		}

	}
	else {
		cout << "Invalid Input!!!\n";
	}

	return 0;
}
