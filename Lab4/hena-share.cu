%%cu
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


void saveFile(double *g, ll matrixSize) {
    ofstream out("finalTemperatures.csv");

    for(int i=0; i<matrixSize; i++) {
        for(int j=0; j<matrixSize; j++) {
            out<<g[i*matrixSize+j]<<',';
        }
        out<<'\n';
    }
}

__global__ void calculateGValue(double *hd, double *gd, ll matrixSize)
{
// Calculate the column index of the g mesh element, denote by x
    // Calculate the row index of the g mesh element, denote by y
    // each thread computes one element of the output matrix Pd.
    // write back to the global memory
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
 
    x = x==0 ? x+1 : x==matrixSize-1 ? x-1 : x;
    y = y==0 ? y+1 : y==matrixSize-1 ? y-1 : y;
 
    gd[x*matrixSize+y] = 0.25*(hd[(x-1)*matrixSize+y] + hd[(x+1)*matrixSize+y] + hd[(x*matrixSize)+y-1] + (hd[x*matrixSize+(y+1)]));
    printf("inside function:: %d %d %f", x, y, gd[x*matrixSize+y]);
}


void jacobiIteration(double *h, double *g, ll matrixSize, ll iterations)
{
    ll size = matrixSize*matrixSize*sizeof(double);
    double *hd, *gd;

    cudaEvent_t start, stop;
    HANDLE_ERROR( cudaEventCreate(&start));
    HANDLE_ERROR( cudaEventCreate(&stop));
    HANDLE_ERROR( cudaEventRecord(start, 0));

    HANDLE_ERROR( cudaMallocManaged((void**)&hd, size));
    HANDLE_ERROR( cudaMallocManaged((void**)&gd, size));
    HANDLE_ERROR( cudaMemcpy(hd, h, size, cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMemcpy(gd, g, size, cudaMemcpyHostToDevice));

    // kernel invocation code
    dim3 dimBlock(4, 4);
    dim3 dimGrid(matrixSize/4, matrixSize/4);
    for (ll iterator = 0; iterator < iterations; iterator++)
    {

        calculateGValue<<<dimGrid, dimBlock>>>( hd, gd, matrixSize);
     /*printf("after function call 1:: %f\n", hd[0]);
     printf("after function call 2:: %f\n", hd[1]);
     printf("after function call 3:: %f\n", hd[2]);*/
      HANDLE_ERROR(cudaDeviceSynchronize());

        for (int y = 1; y < matrixSize-1; y++)
        {
            for (int x = 1; x < matrixSize-1; x++)
            {
                hd[y*matrixSize+x] = gd[y*matrixSize+x];
            }
        }
        
        //HANDLE_ERROR( cudaMemcpy(h,hd,size,cudaMemcpyDeviceToHost));
        //HANDLE_ERROR(cudaGetLastError());
    }

      HANDLE_ERROR( cudaMemcpy(h,hd,size,cudaMemcpyDeviceToHost));
 
 cout<<"final result"<<endl;
      for (int y = 0; y < matrixSize; y++)
      {
          for (int x = 0; x < matrixSize; x++)
          {
              cout<<h[y*matrixSize+x]<<" ";
          }
          cout<<endl;
      }
    HANDLE_ERROR( cudaEventRecord(stop, 0));
    HANDLE_ERROR( cudaEventSynchronize(stop));
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, start, stop));
    cout<<elapsedTime<<endl;

    HANDLE_ERROR( cudaFree(hd));
    HANDLE_ERROR( cudaFree(gd));

    HANDLE_ERROR( cudaEventDestroy(start));
    HANDLE_ERROR( cudaEventDestroy(stop));

}


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
        }
    }


    for(ll i = 0; i < matrixSize; i++) {
        h[i*matrixSize + 0] = 20.0;
        h[i*matrixSize + matrixSize-1] = 20.0;

        h[0*matrixSize + i] = 20.0;
        h[(matrixSize-1)*matrixSize + i] = 20.0;
    }

    for(ll i = (3*matrixSize)/10; i < (7*matrixSize)/10; i++) {
        h[0*matrixSize + i] = 100.0;
    }
 
    for (int y = 0; y < matrixSize; y++)
    {
        for (int x = 0; x < matrixSize; x++)
        {
            cout<<h[y*matrixSize+x]<<" ";
        }
        cout<<endl;
    }

    jacobiIteration(h, g, matrixSize, iterations);

    free(h);
    free(g);
}


void readInput(ll argc, char *argv[], ll &matrixSize, ll &iterations) {
    string argument;

    for(ll i=0; i < argc; i++) {
        string temp(argv[i]);
        argument += temp;
    }

    if(argument.find("-I") != string::npos) {
        ll pos = argument.find("-I")+2;
        iterations = 0;
        while(pos < argument.size() && argument[pos] >= '0' && argument[pos] <= '9') {
            iterations = iterations*10 + argument[pos]-'0';
            pos++;
        }
    }

    if(argument.find("-n") != string::npos) {
        ll pos = argument.find("-n")+2;
        matrixSize = 0;
        while(pos < argument.size() && argument[pos] >= '0' && argument[pos] <= '9') {
            matrixSize = matrixSize*10 + argument[pos]-'0';
            pos++;
        }
    }

    //cout<<iterations<<" "<<matrixSize<<endl;
}


int main()
{
    /*int argc=5;
   // char *argv[] = "minnet -n 32 -I 1";
      //if (argc < 5 || argc > 5)
      //{
          //cout << "Invalid Input!\n";
          //return 0;
      //}*/

    ll matrixSize=6, iterations=1;
    //Taking user input to check for valid values for the unique path calculation
    //readInput(argc, argv, matrixSize, iterations);


    if(matrixSize == -1 || iterations == -1) {
        cout<<"Invalid Input!\n";
        return 0;
    }

    initPlateTemp(matrixSize, iterations);


    return 0;
}