#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>

// Full attention score pipeline

/// @brief Calculates attention scores
/// @param Q Query matrix
/// @param KT Transposed Key matrix
/// @param Score Score matrix for output
/// @param N Query matrix width
/// @param M Key matrix width
/// @param d Matrix height (Query and Key)
void calculateAttentionScores(const float* Q, const float* KT, float* Score, int N, int M, int d)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < d; k++)
            {
                sum += Q[i * d + k] * KT[k * M + j];
            }

            Score[i * M + j] = sum;
        }
    }
}


/// @brief Transpose input matrix into secondary matrix
/// @param K Source matrix pointer
/// @param KT Destination matrix pointer
/// @param M Matrix (source) height
/// @param d Matrix (source) width
void transpose(const float* K, float* KT, int M, int d)
{
    for (int i = 0; i < M; i++)
    {
        for (int k = 0; k < d; k++)
        {
            KT[k * M + i] = K[i * d + k];
        }
    }
}


/// @brief Calculate softmax scores (by row)
/// @param Score Score matrix pointer
/// @param N Number of rows
/// @param M Number of columns
void softmaxRow(float* Score, int N, int M)
{
    // Row
    for (int i = 0; i < N; ++i)
    {
        // Get max value from row
        float maxValue = Score[i * M];

        // Col
        for (int j = 1; j < M; ++j)
        {
            if (Score[i * M + j] > maxValue)
                maxValue = Score[i * M + j];
        }

        float sum = 0.0f;

        // Col
        for (int j = 0; j < M; ++j)
        {
            // Copy numeric stability technique from PyTorch (prevents overflows)
            // Subtracting guarantees that the exponentiations result in at most 1.
            float e = expf(Score[i * M + j] - maxValue);

            // Set score to exponent solution for numerator
            Score[i * M + j] = e;

            // Sum for denominator
            sum += e;
        }

        // Calc denominator
        float inv = 1.0f / sum;

        // Calculate softmax in-place
        for (int j = 0; j < M; ++j)
        {
            Score[i * M + j] *= inv;
        }
    }
}


/// @brief Scale scores with scaling factor
/// @param Score Scores pointer
/// @param N Number of rows
/// @param M Number of cols
/// @param scale Scaling factor
void scaleScore(float* Score, int N, int M, float scale)
{
    int valCount = N * M;

    for (int i = 0; i < valCount; ++i)
    {
        Score[i] *= scale;
    }
}


int main(int argc, char** argv)
{
    int N = 1024;
    int M = 1024;
    int d = 64;

    int averageOver = 1;
    int consoleOutput = 1;

    if (argc >= 4)
    {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        d = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        averageOver = atoi(argv[4]);
    }
    if (argc >= 6)
    {
        consoleOutput = atoi(argv[5]);
    }

    float* Q = (float*)malloc((size_t)N * d * sizeof(float));
    float* K = (float*)malloc((size_t)M * d * sizeof(float));
    float* KT = (float*)malloc((size_t)d * M * sizeof(float));
    float* score = (float*)malloc((size_t)N * M * sizeof(float));
    double* timeList = (double*)malloc((size_t)averageOver * sizeof(double));

    if (!Q || !K || !KT || !score || !timeList)
    {
        fprintf(stderr, "Allocation failed\n");
        free(Q);
        free(K);
        free(KT);
        free(score);
        free(timeList);
        return 1;
    }

    if (consoleOutput)
    {
        int64_t arithmeticIntensityFlops = (int64_t)2 * N * M * d + (int64_t)N * M + (int64_t)4 * N * M;
        int64_t bytesMoved = (int64_t)2 * M * d * 4 + (int64_t)4 * (N * d + M * d + N * M) + (int64_t)2 * N * M + (int64_t)2 * N * M * 4;
        double arithmeticIntensity = (double)arithmeticIntensityFlops / (double)bytesMoved;

        printf("Arithmetic Intensity=%f\n", arithmeticIntensity);
        printf("Bytes Moved=%lld\n", (long long)bytesMoved);
    }

    // Fill with defaults
    for (int i = 0; i < N * d; ++i)
        Q[i] = 0.5f;
    for (int i = 0; i < M * d; ++i)
        K[i] = 0.5f;
    for (int i = 0; i < N * M; ++i)
        score[i] = 0.0f;

    for (int i = 0; i < averageOver; ++i)
    {
        transpose(K, KT, M, d);

        clock_t start = clock();

        calculateAttentionScores(Q, KT, score, N, M, d);
        float scaleF = 1.0f / sqrtf((float)d);
        scaleScore(score, N, M, scaleF);
        softmaxRow(score, N, M);

        clock_t end = clock();
        timeList[i] = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    }

    if (consoleOutput)
    {
        double time = 0.0;
        for (int i = 0; i < averageOver; ++i)
            time += timeList[i];
        time /= averageOver;

        int64_t arithmeticIntensityFlops = (int64_t)2 * N * M * d + (int64_t)N * M + (int64_t)4 * N * M;
        int64_t bytesMoved = (int64_t)2 * M * d * 4 + (int64_t)4 * (N * d + M * d + N * M) + (int64_t)2 * N * M + (int64_t)2 * N * M * 4;
        double arithmeticIntensity = (double)arithmeticIntensityFlops / (double)bytesMoved;

        printf("Time: %f ms\n", time);
        printf("Arithmetic Intensity=%f\n", arithmeticIntensity);
        printf("Bytes Moved=%lld\n", (long long)bytesMoved);
    }

    free(Q);
    free(K);
    free(KT);
    free(score);
    free(timeList);

    return 0;
}
