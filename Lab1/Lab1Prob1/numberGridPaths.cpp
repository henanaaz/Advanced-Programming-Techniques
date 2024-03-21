/*
Author: Hena Naaz
Class: ECE6122 (A)
Last Date Modified: 26 Sept
Description: Lab1Prob1
Purpose of this file: a function contained in a separate implementation file (numberGridPaths.cpp) 
To calculate the total number of paths using the function prototype show below:
	uint64_t numberGridPaths(unsigned int nRows, unsigned int nCols);
*/

#include <vector>

/**
 * Function numberGridPaths() takes rows and columns as arguments 
 * and calculates the number of unique paths are possible for different sized grid mazes.
 * 
 * Time Complexity:  O(rows*columns)
 * Space Complexity: O(columns)
 * 
 * @param nRows unsigned integer total rows in matrix
 * @param nCols unsigned integer total columns in matrix
 * @return unsigned integer total possible paths to reach the target 
 */
uint64_t numberGridPaths(unsigned int nRows, unsigned int nCols) 
{
	// The vector prevRow is to store the previous computed unique path till (i-1)th row.
	// Initially, first row has only one unique path
	vector<uint64_t> prevRow(nCols, 1);

	// For loop to go over all the rows, by moving one step down the grid maze in each iteration.
	for (uint64_t iRow = 1; iRow < nRows; iRow++)
		{
		// For loop to go over all the columns, by moving one step right in the grid maze in each row.
		for (uint64_t jCol = 1; jCol < nCols; jCol++)
		{
			// dp[iRow][jCol] = dp[iRow-1][jCol]+dp[iRow][jCol-1]
			prevRow[jCol] = prevRow[jCol] + prevRow[jCol - 1];
		}
	}

	return prevRow[nCols - 1];
}



















