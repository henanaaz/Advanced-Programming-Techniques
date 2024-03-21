/*
Author: Jeff Hurley
Class: ECE4122 or ECE6122 (all sections)
Last Date Modified: 9/26/2022

Description:

Generates a text file contaning two matrices to be multiplied together

*/


#include <fstream>
#include <random>

int main()
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 10.0);

    std::fstream fs("SampleMatrixFile90.txt", std::fstream::out);

    int m(90), n(90), p(90);

    fs << m << " " <<n << "\n";

    for (int ii = 0; ii < m; ii++)
    {
        for (int jj = 0; jj < n-1; jj++)
        {
            fs << distribution(generator) << " ";
        }
        fs << distribution(generator) << "\n";
    }

    fs << n << " " << p << "\n";

    for (int ii = 0; ii < n-1; ii++)
    {
        for (int jj = 0; jj < p - 1; jj++)
        {
            fs << distribution(generator) << " ";
        }
        fs << distribution(generator) << "\n";
    }
    for (int jj = 0; jj < p - 1; jj++)
    {
        fs << distribution(generator) << " ";
    }
    fs << distribution(generator);

    fs.close();
}
