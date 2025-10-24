#include "../include/tensor.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

void addTensor(Tensor *A, Tensor *B, Tensor *Result) {
    if (!A || !B || !Result) {
        fprintf(stderr, "Null tensor pointer!\n");
        exit(1);
    }

    Result->ndim = A->ndim;

    if (!Result->shape)
        Result->shape = malloc(sizeof(int) * Result->ndim);

    int size = 1;
    for (int i = 0; i < A->ndim; i++) {
        size*=A->shape[i];
        Result->shape[i] = A->shape[i];
    }

    if (!Result->data)
        Result->data = malloc(sizeof(float) * size);

    for (int i =0; i < size; i++) {
        Result->data[i] = A->data[i] + B->data[i];
    }
}

void matmulTensor(Tensor *A, Tensor *B, Tensor *Result) {
    if (!A || !B || !Result || A->ndim!=2 || B->ndim!=2 || A->shape[1]!=B->shape[0]) {
        fprintf(stderr, "bad matrix!\n");
        exit(1);
    }
    Result->ndim = 2;

    if (!Result->shape)
        Result->shape = malloc(sizeof(int) * Result->ndim);
    Result->shape[0] = A->shape[0];
    Result->shape[1] = B->shape[1];

    int size = 1;
    for (int i = 0; i < A->ndim; i++)
        size*=A->shape[i];
    if (!Result->data)
        Result->data = malloc(sizeof(float) * size);

    // basic
    for (int i = 0; i < A->shape[0]; i++) {
        for (int j = 0; j < B->shape[1]; j++) {
            Result->data[i*Result->shape[1] + j] = 0;
            for (int k = 0; k < A->shape[1]; k++) {
                Result->data[i*Result->shape[1] + j]+=A->data[i*A->shape[1]+k] * B->data[k*B->shape[1]+j];
            }
        }
    }
}

void scaleTensor(Tensor *A, float k) {
    int size = 1;
    for (int i = 0; i < A->ndim; i++)
        size*=A->shape[i];

    for (int i = 0; i < size; i++) {
        A->data[i] *= k;
    }
} 

void transposeTensor(Tensor *A, Tensor *Result) {
    Result->ndim = A->ndim;
    if (!Result->shape)
        Result->shape = malloc(sizeof(int) * Result->ndim);
    Result->shape[0] = A->shape[1];
    Result->shape[1] = A->shape[0];

    int size = 1;
    for (int i = 0; i < A->ndim; i++)
        size*=A->shape[i];
    if (!Result->data)
        Result->data = malloc(sizeof(float) * size);

    for (int i = 0; i < A->shape[0]; i++) {
        for (int j = 0; j < A->shape[1]; j++) {
            Result->data[j*Result->shape[1] + i] = A->data[i*A->shape[1] + j];
        }
    }
}

void softmaxTensor(Tensor *A) {
    for (int i = 0; i < A->shape[0]; i++) {
        float max = -INFINITY;
        float sum = 0;
        for (int j = 0; j < A->shape[1]; j++) {
            float val = A->data[i*A->shape[1]+j];
            max = (max > val) ? max : val;
        }
        for (int j = 0; j < A->shape[1]; j++) {
            float val = expf(A->data[i*A->shape[1]+j] - max);
            A->data[i*A->shape[1]+j] = val;
            sum+=val;            
        }
        for (int j = 0; j < A->shape[1]; j++) {
            A->data[i*A->shape[1]+j]/=sum;
        }
    }
}

void addBiasToTensor(Tensor *A, float *Bias) {
    for (int i = 0; i < A->shape[0]; i++) {
        for (int j = 0; j < A->shape[1]; j++) {
            A->data[i*A->shape[1]+j] += Bias[j];
        }
    }
}

void copyTensor(Tensor *A, Tensor *Result) {
    freeTensor(Result);
    Result->ndim = A->ndim;
    Result->shape = malloc(sizeof(int) * Result->ndim);
    int size = 1;
    for (int i = 0; i < Result->ndim; i++) {
        Result->shape[i] = A->shape[i];
        size*=A->shape[i];
    }
    Result->data = malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++) {
        Result->data[i] = A->data[i];
    }
}

void freeTensor(Tensor *A) {
    if (A->shape) free(A->shape);
    if (A->data) free(A->data);
    A->shape = NULL;
    A->data = NULL;
    A->ndim = 0;
}

void printTensor(Tensor *A) {
    for (int i = 0; i < A->shape[0]; i++) {
        for (int j = 0; j < A->shape[1]; j++) {
            printf("%f ", A->data[i*A->shape[1]+j]);
        }
        printf("\n");
    }
}