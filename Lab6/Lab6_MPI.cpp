/*
Author: Hena Naaz
Class: ECE6122
Last Date Modified: 3 December 2022
Description: Lab 6 for APT Fall 2022
What is the purpose of this file?
Simulate the steady state 2D problem with Jacobi iteration and using OpenMPI to achieve parallelism in the program.
The code also has some debug codes that are commented out for debug purposes.
*/

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "mpi.h"

#define ll long long int

using namespace std;

/*
Function to read the input and check for invalid conditions
*/
void readInput(ll argc, char *argv[], ll &width, ll &iterations) {
    string argument;

    for(ll i=0; i < argc; i++) 
    {
        string temp(argv[i]);
        argument += temp;
    }

	//Calculate number of Iterations : string to int conversion
    if(argument.find("-I") != string::npos) 
    {
        ll pos = argument.find("-I")+2;
        iterations = 0;
        while(pos < argument.size() && argument[pos] >= '0' && argument[pos] <= '9')
        {
            iterations = iterations*10 + argument[pos]-'0';
            pos++;
        }
    }

	//Calculate number of interior points provided : string to int conversion
    if(argument.find("-n") != string::npos)
    {
        ll pos = argument.find("-n")+2;
        width = 0;
        while(pos < argument.size() && argument[pos] >= '0' && argument[pos] <= '9')
        {
            width = width*10 + argument[pos]-'0';
            pos++;
        }
        
        if(width > -1)width = width +2;
        
    }
    


}

int main(int argc, char** argv)
{

	// Creating variabled for the program
	int rank, size, ii, jj, kk; // iterator and processor variables
	int index;
	ll numIterations=-1;
	ll width=-1;
    
    //Check arguments and provided to the file
    readInput(argc, argv, width, numIterations);

    
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Invalid check done by the server with rank = 0
    if ( rank == 0 )
    {    
        if (argc < 5 || argc > 5)
        {
            cout << "Invalid Input1!\n";
            cout << argc << endl;
            return 0;
        }

        //Taking user input to check for valid values for the unique path calculation
        //readInput(argc, argv, width, numIterations);

        if(width == -1 || numIterations == -1) 
        {
            cout<<"Invalid Input2!\n";
            return 0;
        }
        
    }

    double t1, t2; 
    t1 = MPI_Wtime(); 


	//We need to allocate oversize arrays so that we can send the same amount of data to each processor
	int newWidth = size * (int)std::ceil((double)width/(double)size);
	int send_count = (newWidth * newWidth) /size;
	int recv_count = send_count;
	double* pPrevious = new double[newWidth * newWidth];
	double* pNext = new double[newWidth * newWidth];
	//cout<<width<<"with:newWidth"<<newWidth<<endl;

	//Determine the number of rows each processor will get
	int block_size = (newWidth/size);

	//Have all the processors initialize their arrays
	for(ii = 0; ii<width; ii++)
	{
		for(jj=0; jj<width; jj++)
		{
			if(ii==0 || ii==width-1 || jj==0 || jj==width-1)
			{			
			    index = ii * newWidth + jj;
			
				pNext[index] = 20.0;
				pPrevious[index] = 20.0;
				//if(rank==0) cout<<index<<"index" <<pPrevious[index] <<endl;
			}
			else
			{		
				// Initialize all interior points as 0.0 
				index = ii * newWidth + jj;
				pNext[index] = 0.0;
				pPrevious[index] = 0.0;
			}
		}
	}

	//Initialize the boundary for hot thin plate
	for(ii=(3*width)/10; ii<(7*width)/10; ii++)
	{
	    if(ii>=(ceil(0.3*width)) && (ii<=floor(0.7*width)-1))
	    {
		pNext[0*newWidth + ii] = 100.0;
	    	pPrevious[0*newWidth + ii] = 100.0;
	    }
	}
	
	int startHere = rank*block_size;

	//Perform Jacobi Iteration for steady state temperature
	for(kk=0; kk<numIterations; kk++)
	{
		for(ii=startHere; ii<startHere+block_size; ii++)
		{
			if(ii==0 || ii>=(width -1))  //oversized 2D array implementation so skipping exterior points
				continue;
			for(jj=0; jj<width; jj++)
			{
				//Skip the first and last rows as they are exterior points
				if (jj==0 || jj>=(width -1))
					continue;
				index = ii * newWidth + jj;
				//performing jacobi Iteration
				pNext[index] = 0.25*(pPrevious[(ii-1)*newWidth + jj] + pPrevious[(ii+1)*newWidth + jj] + pPrevious[(ii*newWidth)+jj-1] + pPrevious[(ii*newWidth) + jj+1]);
			}
		}

		MPI_Allgather( &pNext[startHere * newWidth], send_count, MPI_DOUBLE, pPrevious,	recv_count, MPI_DOUBLE, MPI_COMM_WORLD);
	}


    t2 = MPI_Wtime(); 

	// Writing the array to the file
	if(rank == 0)
	{
		// Server/rank=0 prints the total time taken
		printf( "Thin plate calculation took %f milliseconds.\n", (t2 - t1)*1000 );
		    
		ofstream outfile;
		outfile.open("finalTemperatures.csv");

		// Write the output file with exterior and interior points after steady state run as per iterations provided
		for(int i=0; i<width; i++)
		{
			for(int j=0; j<width; j++)
			{
				outfile<<std::fixed<<std::setprecision(15) << pPrevious[i*newWidth+j] << ",";
				//cout<<i*newWidth+j<<"=index"<<std::fixed<<std::setprecision(15) << pPrevious[i*newWidth+j] << "\n";
			}
			outfile << "\n";
			//cout<<"\n";
		}
		outfile.close();
	}

	MPI_Finalize();
	return 0;

}
