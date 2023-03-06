/*
This file provides matvec implementation using the macOS Accelerate
Framework, which seems to be the most optimized matrix matrix multiplication
on macOS.
*/
#include <Accelerate/Accelerate.h>

void acc_sgemm(int m, int n, int k, float *A, float *B, float *C) {
    //A[m][k]
    //B[k][n]
    //C[m][n]
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, m, B, k, 0.0, C, m);
}

void acc_sgemm_t(int m, int n, int k, float *A, float *B, float *C) {
    //A[k][m] (to be transposed)
    //B[k][n]
    //C[m][n]
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, k, 0.0, C, m);
}
