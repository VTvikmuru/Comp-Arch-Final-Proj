#include <vector>
#include <chrono>
#include <iostream>
#include <cmath>

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
            float sum = 0;

            for (int k = 0; k < d; k++) 
            {
                sum += Q[i * d + k] * KT[k * M + j];
            }

            Score[i * M + j] = sum;
        }
    }
}


/// @brief Transpose input matix into secondary matrix
/// @param K Source matrix pointer
/// @param KT Destination matrix pointer
/// @param L Matrix (source) height
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
        // Get mex value from row
        float maxValue = Score[i * M];

        // Col
        for (int j = 1; j < M; ++j)
        {
            if (Score[i * M + j] > maxValue)
                maxValue = Score[i * M + j];
        }

        float sum = 0;

        // Col
        for (int j = 0; j < M; ++j)
        {
            // Copy numeric stability technique from PyTorch (prevents overflows)
            // Subtracting guarantees that the exponentiations result in at most 1. (wikipedia.org Softmax_function)
            float e = std::exp(Score[i * M + j] - maxValue);

            // Set score to exponent solution for numerator
            Score[i * M + j] = e;

            // Sum for denominator
            sum += e;
        }

        // Calc denominator
        float inv = 1 / sum;

        //  Calculate softmax in-place
        for (int j = 0; j < M; ++j)
        {
            Score[i * M + j] = Score[i * M + j] * inv;
        }
    }
}

/// @brief Scale scores with scaling factor
/// @param Score SCores pointer
/// @param N Number of rows
/// @param M Number of cols
/// @param scale Scaling factor
void scaleScore(float* Score, int N, int M, float scale) 
{
    int valCount = N * M;

    for (int i = 0; i < valCount; ++i) 
    {
        Score[i] = Score[i] * scale;
    }
}

int main(int argc, char** argv) {
    int N = 1024;
    int M = 1024;
    int d = 64;
    

    int averageOver = 1;
    int consoleOutput = 1;

    if (argc >= 4) 
    {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        d = std::atoi(argv[3]);
    }
    if (argc >= 5)
    {
        averageOver = std::atoi(argv[4]);
    }
    if (argc >= 6)
    {
        consoleOutput = std::atoi(argv[5]);
    }

    std::vector<float> Q(N * d);
    std::vector<float> K(M * d);
    std::vector<float> score(N * M);

    // Fill with defaults
    for (auto& x : Q) x = 0.5f;
    for (auto& x : K) x = 0.5f;

    // Transpose K
    std::vector<float> KT(d * M);
    for (int i = 0; i < M; i++)
    {
        for (int k = 0; k < d; k++)
        {
            KT[k * M + i] = K[i * d + k];
        }
    }

    K = KT;
    
    std::vector<double>timeList;

    for (int i = 0; i < averageOver; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        transpose(K.data(), KT.data(), M, d);
        calculateAttentionScores(Q.data(), KT.data(), score.data(), N, M, d);
        
        float scaleF = 1 / std::sqrt((float)d);

        scaleScore(score.data(), N, M, scaleF);
        softmaxRow(score.data(), N, M);

        auto end = std::chrono::high_resolution_clock::now();
        timeList.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    
    if(consoleOutput)
    {
        // Average time
        double time = 0;
        for (int i = 0; i < timeList.size(); i++)
        {
            time += timeList[i];
        }
        time = time / timeList.size();
        
        // TODO: Calculating arithmetic intensity
        // Transpose arithmetic intensity = 0
        // Attention score arithmetic intensity = 2NMd
        // Scaling arithmetic intensity = NM
        // Row softmax arithmetic intensity = N * 4M = 4NM
        int arithmeticIntensityFlops = 2 * N * M * d + N * M + 4 * N * M;
        
        // Transpose bytes moved = 2 * Md * 4
        // Attention score bytes moved = 2 * GEMM = 4 * (Nd + Md + NM)
        // SCaling bytes moved = 2NM
        // Softmax bytes moved = 2NM * 4
        int airthmeticIntensityBytesMoved = 2 * M * d * 4 + 4 * (N * d + M * d + N * M) + 2 * N * M + 2 * N * M * 4;

        std::cout << "Time: " << time << " ms" << std::endl;
        std::cout << "Arithmetic Intensity=" << arithmeticIntensityFlops/airthmeticIntensityBytesMoved << std::endl;
        std::cout << "Bytes Moved=" << airthmeticIntensityBytesMoved << std::endl;

    }

}
