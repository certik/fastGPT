/*
This file provides matvec implementation using OpenBLAS.
*/
#include <cblas.h>

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
    printf("acc_sgemm_t: %d %d %d\n", m, n, k);
    // TODO: A seems correct, but B[0] is incorrect:
    printf("Values: %f %f %f\n", A[0], B[0], C[0]);
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, k, 0.0, C, m);
    printf("BLAS done\n");
}
