/*
Author: Hena Naaz
Class: ECE6122
Last Date Modified: 10 November 2022
Description: Lab 4 for APT Fall 2022
What is the purpose of this file?
Simulate the steady state 2D problem using Jacobi iteration and using CUDA to achieve parallelism in the program.
*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <assert.h>

using namespace std;

#define ll long long int


/*
Handle error for CUDA functions
*/
inline cudaError_t HANDLE_ERROR(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess)
  {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
    return result;
}

/*
Function to save file in csv file
*/
void saveFile(double *g, ll matrixSize) {
    ofstream out("finalTemperatures.csv");

    for(int i=0; i<matrixSize; i++) 
    {
        for(int j=0; j<matrixSize; j++) 
        {
            out<<g[i*matrixSize+j]<<',';
        }
        out<<'\n';
    }
}


/*
Function to perform matrix manipulation for steady state using CUDA kernels
*/
__global__ void calculateGValue(double *hd, double *gd, ll matrixSize)
{
// Calculate the column index of the g mesh element, denote by x
    // Calculate the row index of the g mesh element, denote by y
    // each thread computes one element of the output matrix Pd.
    // write back to the global memory
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
 
    if((x>0 && x<matrixSize-1) &&(y>0 && y<matrixSize-1))
    {
        gd[y*matrixSize+x] = 0.25*(hd[(y-1)*matrixSize+x] + hd[(y+1)*matrixSize+x] + hd[(y*matrixSize)+x-1] + (hd[y*matrixSize+(x+1)]));
    }
}

/*
Function to perform cudo copy and start jacobi iterations
*/
void jacobiIteration(double *h, double *g, ll matrixSize, ll iterations)
{
    ll size = matrixSize*matrixSize*sizeof(double);
    double *hd, *gd;

    cudaEvent_t start, stop;
    HANDLE_ERROR( cudaEventCreate(&start));
    HANDLE_ERROR( cudaEventCreate(&stop));
    HANDLE_ERROR( cudaEventRecord(start, 0));

    HANDLE_ERROR( cudaMalloc((void**)&hd, size));
    HANDLE_ERROR( cudaMalloc((void**)&gd, size));
    HANDLE_ERROR( cudaMemcpy(hd, h, size, cudaMemcpyHostToDevice));
 
    // kernel invocation code
    int numThreads = 32; //for faster calculations
    dim3 dimBlock(numThreads, numThreads);
    dim3 dimGrid((matrixSize+numThreads-1)/numThreads, (matrixSize+numThreads-1)/numThreads);
    
    //Initialize the device copy of gd
    gd = hd;
    for (ll iterator = 0; iterator < iterations; iterator++)
    {
   	calculateGValue<<<dimGrid, dimBlock>>>( hd, gd, matrixSize);
	hd = gd;
    }
    HANDLE_ERROR( cudaMemcpy(h,hd,size,cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    HANDLE_ERROR( cudaEventRecord(stop, 0));
    HANDLE_ERROR( cudaEventSynchronize(stop));
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, start, stop));
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );
    saveFile(h, matrixSize);
    HANDLE_ERROR( cudaFree(hd));
    HANDLE_ERROR( cudaFree(gd));

    HANDLE_ERROR( cudaEventDestroy(start));
    HANDLE_ERROR( cudaEventDestroy(stop));

}

/*
Function to initialize the matrix with plate and interiors points with 0,20 and 100 temperature values
*/
void initPlateTemp(ll matrixSize, ll iterations)
{
    ll size = matrixSize*matrixSize*sizeof(double);
    double *h = (double *) malloc(size);
    double *g = (double *) malloc(size);

    // initialize the matrices
    for (int y = 0; y < matrixSize; y++)
    {
        for (int x = 0; x < matrixSize; x++)
        {
            h[y*matrixSize + x] = 0;
            g[y*matrixSize + x] = 0;
        }
    }


    for(ll i = 0; i < matrixSize; i++) 
    {
        h[i*matrixSize + 0] = 20.0;
        g[i*matrixSize + 0] = 20.0;
        h[i*matrixSize + matrixSize-1] = 20.0;
        g[i*matrixSize + matrixSize-1] = 20.0;

        h[0*matrixSize + i] = 20.0;
        g[0*matrixSize + i] = 20.0;
        h[(matrixSize-1)*matrixSize + i] = 20.0;
        g[0*matrixSize + i] = 20.0;
    }

    for(ll i = (3*matrixSize)/10; i < (7*matrixSize)/10; i++)
    {
        h[0*matrixSize + i] = 100.0;
        g[0*matrixSize + i] = 100.0;
    }
 
    jacobiIteration(h, g, matrixSize, iterations);

    saveFile(h, matrixSize);
    free(h);
    free(g);
}

/*
Function to read the input and check for invalid conditions
*/
void readInput(ll argc, char *argv[], ll &matrixSize, ll &iterations) {
    string argument;

    for(ll i=0; i < argc; i++) 
    {
        string temp(argv[i]);
        argument += temp;
    }

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

    if(argument.find("-n") != string::npos)
    {
        ll pos = argument.find("-n")+2;
        matrixSize = 0;
        while(pos < argument.size() && argument[pos] >= '0' && argument[pos] <= '9')
        {
            matrixSize = matrixSize*10 + argument[pos]-'0';
            pos++;
        }
    }

}


int main( int argc, char *argv[])
{
    
        if (argc < 5 || argc > 5)
        {
                cout << "Invalid Input!\n";
                return 0;
        }

        ll matrixSize=-1, iterations=-1;
        //Taking user input to check for valid values for the unique path calculation
        readInput(argc, argv, matrixSize, iterations);


        if(matrixSize == -1 || iterations == -1) 
        {
                cout<<"Invalid Input!\n";
                return 0;
        }

        const int matrixSize2 = matrixSize + 2;

        initPlateTemp(matrixSize2, iterations);


        return 0;

}
