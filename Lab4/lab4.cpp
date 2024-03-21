/*
Author: Hena Naaz
Class: ECE6122
Last Date Modified: 7 November 2022
Description: Lab 4 for APT Fall 2022
What is the purpose of this file?
Simulate the steady state 2D problem using Jacobi iteration and using CUDA to achieve parallelism in the program.
*/

#include <iostream>
#include <fstream>

using namespace std;

/*
Function initPlateTemp() is responsible for setting up the boundary conditions for the thin plate on all 4 sides
Simulate the following on the mesh:
A perfectly insulted thin plate with the sides held at 20 °C and a short segment on one side is held at 100 °C
Sides = 10ft
segment = 4ft (on only one side)
*/
void initPlateTemp(double initMesh[int row, int col])
{
	for (int i = 1; i < n; i++)
	{
		for (int j = 1; j < n; j++)
		{
			h[1][j] = 100;
			h[i][1] = 100;
			h[n][j] = 100;
			h[i][n] = 100;

		}
		
		for (int j = (n / 3); j < (2n / 3); j++)
		{
			h[0][j] = 20;
		}

	}
}

/*
Function jacobiIteration1() deals with the first step 
*/
int jacobiIteration1(int iteration)
{
	double h;
	double g;
	for ( iteration = 0; iteration < limit; iteration++)
	{
		for (int i = 1; i < n; i++)
		{
			for (int j = 1; j < n; j++)
			{
				g[i][j] = 0.25 * (h[i - 1][j] + h[i + 1][j] + h[i][j - 1] + h[i][j + 1]);
			}
		}

		for (int i = 1; i < n; i++)
		{
			for (int j = 1; j < n; j++)
			{
				h[i][j] = g[i][j];
			}
		}
	}
}

/*
Function jacobiIteration2() deals with the second step
*/
int jacobiIteration2()
{

}

int main()
{
	if (argc == )
	if (argc < 3 || argc > 3 || rows == -1 || cols == -1)
	{
		cout << "Invalid Input!\n";
		return 0;
	}

	int64_t rows = 0;
	int64_t cols = 0;

	//Taking user input to check for valid values for the unique path calculation
	readInput(argc, argv, rows, cols);

	double finalTemp = 0.0;
	ofstream myOutFile("finalTemperatures.txt", ofstream::out);
	myOutFile << finalTemp << endl;


	// writing the number of unique paths from function numberGridPaths() to the output file
	double latency = steadyStateProblem(rows, cols);
	cout << "Thin plate calculation took %d miliseconds." << latency << endl;

	myOutFile.close();

	return 0;


}