/*
Author: Hena Naaz
Class: ECE6122 A
Last Date Modified: 12 September 2022
Description: Lab 0 Problem 2
What is the purpose of this file?
Purpose:to solve the problem 2 -"One more One" 
(count number of ones added when divided by 7)
*/

#include <iostream>
using namespace std;

/* Function countOnes(n):
Creating a function called countOnes() to calculate the number of numbers
The function takes the user input 'n' to calculate the number of instances of 1, under that number 'n' when divided by 7
Function countOnes() then returns the number of instances of ones through variable 'result'
*/
int countOnes(int n) {
	int result = 0;

	//If n=1, do nothing and the process stops
	if (n == 1)
		return 0;

	//If n is divisible by 7, divide it by 7, otherwise add 1 and increement the number of ones
	while(n != 1) {
		if(n%7 == 0) {
			n /= 7;
		}
		else {
			n++;
			result++;
		}
	}

	return result;
}

int main(void) {
	while(1) {
		int n;
		
		//This part of the program is taking user input from the command line
		cout<<"Please enter the starting number n: ";
		cin>>n;

		//Check if the input is valid integer or not
		if (cin.fail())
		{
			cout << "Invalid input!! Please try again.\n";
			cin.clear(); // reset the failed state
			cin.ignore();
			continue; //continuing with the user input loop
		}

		//Check if the input is a valid positive integer or not
		if(n < 0) {
			cout<<"Invalid input!! Please try again.\n";
			continue;
		}

		if(n == 0)
			return 0;

		//Printing out the number of instances returned by the function countOnes()
		cout<<"The sequence had "<<countOnes(n)<<" instances of the number 1 being added."<<endl;
	}

	return 0;
}