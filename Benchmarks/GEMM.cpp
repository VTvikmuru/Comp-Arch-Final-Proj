// Author: Vikram Muruganandam

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>

inline void GEMM(const double* A, const double* B, double* C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i) 
    {
        for (int j = 0; j < N; ++j) 
        {
            double sum = 0.0;
            for (int p = 0; p < K; ++p) 
            {
                sum += A[i * K + p] * B[p * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

/// @brief Initialize matrixes randomly with the given seed
/// @param Mptr Matrix pointer
/// @param rows Matrix height
/// @param cols Matrics width
/// @param seed Seed for random number generator
void initMatrix(double* Mptr, int rows, int cols, unsigned seed)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for (int i = 0; i < rows * cols; ++i) 
    {
        Mptr[i] = distribution(generator);
    }
}

/// @brief Creates checksum to ensure repeatability and verification of results and multiplication
/// @param C GEMM Product matrix
/// @param M Matrix width
/// @param N Matrix height
/// @return Checksum
double checksum(const double* C, int M, int N)
{
    double sum = 0.0;
    for (int i = 0; i < M * N; ++i) 
    {
        sum += C[i];
    }
    return sum;
}


int main(int argc, char** argv)
{
    // Default sizes - Using square sizes based on NVIDIA Matrix Multiplication Background User Guide
    int M = 256;
    int N = 256;
    int K = 256;
    int iterations = 10;

    // Keep or remove console output and checksum calculation, so that runtime is closer to multiplication time
    bool doChecksum = true;

    // Average over this many runs
    int averageOver = 1;
    std::vector<double>* timeList = new std::vector<double>();
    double time = 0;

    // Parse arguments
    if(argc == 2)
    {
        iterations = std::atoi(argv[1]);
    }
    if (argc == 3)
    {
        
        doChecksum = std::atoi(argv[2]);
    } else {
        if (argc >= 4) 
        {
            M = std::atoi(argv[1]);
            N = std::atoi(argv[2]);
            K = std::atoi(argv[3]);
        }
        if (argc >= 5) 
        {
            iterations = std::atoi(argv[4]);
        }
        if (argc >= 6)
        {
            doChecksum = std::atoi(argv[5]);
        }
        if (argc >= 7)
        {
            averageOver = std::atoi(argv[6]);
        }
    }

    if(doChecksum) 
    {
        std::cout << "GEMM benchmark \n";
        std::cout << "M=" << M << " N=" << N << " K=" << K << " iters=" << iterations << " averageOver=" << averageOver << std::endl;
        std::cout << "Arithmetic Intensity=" << (M * N * K) / (M * K + N * K + M * N) << std::endl;
        std::cout << "Bytes Moved=" << 2 * (M * K + N * K + M * N) << std::endl;
    }

    // Create matrixes
    std::vector<double> A(M * K);
    std::vector<double> B(K * N);
    std::vector<double> C(M * N);

    for (int i = 0; i < averageOver; i++)
    {

        // Init matrixes
        initMatrix(A.data(), M, K, 42);
        initMatrix(B.data(), K, N, 1337);
        initMatrix(C.data(), M, N, 7);

        // Warmup cache - TO avoid cold start effects on performance
        GEMM(A.data(), B.data(), C.data(), M, N, K);

        // Time multiplication only as gem5 measures entire program runtime
        auto start = std::chrono::high_resolution_clock::now();

        for (int it = 0; it < iterations; ++it) 
        {
            GEMM(A.data(), B.data(), C.data(), M, N, K);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        timeList->push_back(elapsed.count());
    }

    
    if(doChecksum)
    {
        // Calculate average time
        for (int i = 0; i < timeList->size(); i++)
        {
            time += (*timeList)[i];
        }
        time = time / averageOver;

        // Check multiplication was completed properly each time
        double sum = checksum(C.data(), M, N);

        std::cout << "Checksum: " << sum << std::endl;
        std::cout << "Elapsed time (s): " << time << std::endl;
    }

    delete timeList;

    return 0;
}
