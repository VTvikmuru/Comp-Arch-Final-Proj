// Author: Vikram Muruganandam

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

void GEMM(const double* A, const double* B, double* C, int M, int N, int K)
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
/// @param cols Matrix width
/// @param seed Seed for random number generator
void initMatrix(double* Mptr, int rows, int cols, unsigned seed)
{
    srand(seed);
    for (int i = 0; i < rows * cols; ++i) 
    {
        Mptr[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;  // Random between -1.0 and 1.0
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
    int doChecksum = 1;

    // Average over this many runs
    int averageOver = 1;
    double* timeList = (double*)malloc(averageOver * sizeof(double));
    double time = 0;

    // Parse arguments
    if(argc == 2)
    {
        iterations = atoi(argv[1]);
    }
    if (argc == 3)
    {
        doChecksum = atoi(argv[2]);
    } else {
        if (argc >= 4) 
        {
            M = atoi(argv[1]);
            N = atoi(argv[2]);
            K = atoi(argv[3]);
        }
        if (argc >= 5) 
        {
            iterations = atoi(argv[4]);
        }
        if (argc >= 6)
        {
            doChecksum = atoi(argv[5]);
        }
        if (argc >= 7)
        {
            averageOver = atoi(argv[6]);
        }
    }

    if(doChecksum) 
    {
        printf("GEMM benchmark \n");
        printf("M=%d N=%d K=%d iters=%d averageOver=%d\n", M, N, K, iterations, averageOver);
        printf("Arithmetic Intensity=%f\n", (double)(M * N * K) / (M * K + N * K + M * N));
        printf("Bytes Moved=%lld\n", 2LL * (M * K + N * K + M * N));
    }

    // Create matrixes
    double* A = (double*)malloc(M * K * sizeof(double));
    double* B = (double*)malloc(K * N * sizeof(double));
    double* C = (double*)malloc(M * N * sizeof(double));

    for (int i = 0; i < averageOver; i++)
    {
        // Init matrixes
        initMatrix(A, M, K, 42);
        initMatrix(B, K, N, 1337);
        initMatrix(C, M, N, 7);

        // Warmup cache - To avoid cold start effects on performance
        GEMM(A, B, C, M, N, K);

        // Time multiplication only as gem5 measures entire program runtime
        clock_t start = clock();

        for (int it = 0; it < iterations; ++it) 
        {
            GEMM(A, B, C, M, N, K);
        }

        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        timeList[i] = elapsed;
    }

    if(doChecksum)
    {
        // Calculate average time
        for (int i = 0; i < averageOver; i++)
        {
            time += timeList[i];
        }
        time = time / averageOver;

        // Check multiplication was completed properly each time
        double sum = checksum(C, M, N);

        printf("Checksum: %f\n", sum);
        printf("Elapsed time (s): %f\n", time);
    }

    free(timeList);
    free(A);
    free(B);
    free(C);

    return 0;
}
