#include "../include/tensor.h"
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>
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

void randomWeights(Tensor *A) {
    srand(time(0));

    int size = A->shape[0] * A->shape[1];
    
    A->data = malloc(sizeof(float) * size);

    float limit = sqrtf(6.0f / (A->shape[0] + A->shape[1]));
    for (int i = 0; i < A->shape[0] * A->shape[1]; i++)
        A->data[i] = ((float)rand() / RAND_MAX) * 2 * limit - limit;

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

void freeTensorData(Tensor *A) {
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



// speed

// we maek highly optimized matrix multiplication
// we use l1 cache :D

// const -> they dont point to same memory
// restrict -> arrays never overlap
// might make compiler optimize better

static inline int prod(const int *shape, int start, int end) {
    int p = 1;
    for (int i = start; i < end; i++)
        p *= shape[i];
    return p;
}

void project_last_axis(const Tensor * restrict A, const Tensor * restrict W, const Tensor * restrict Bias, Tensor * Out) {
    int DimensionOld = A->shape[A->ndim-1];
    int DimensionNew = W->shape[1];
    int B = prod(A->shape, 0, A->ndim-1); // this is the amnt of elements to skip for last axis

    // faster access
    const float * restrict a = A->data;
    const float * restrict w = W->data;
    const float * restrict b = (Bias && Bias->data) ? Bias->data : NULL;
    float * restrict o = Out->data;

    // deos parallel
    #pragma omp parallel for schedule(static)

    // =========================================== ADJUST ===========================================
    // batch skips all the other bs
    for (int batch = 0; batch < B; batch++) {
        const float *a_row = &a[batch * DimensionOld];
        float *o_row = &o[batch * DimensionNew];

        for (int j = 0; j < DimensionNew; j++) {
            // __m256 is a 256 bit register
            __m256 acc = _mm256_setzero_ps();
            float sum = b ? b[j] : 0.0f;
            int k = 0;

            // Vectorized inner product
            for (; k + 8 <= DimensionOld; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(&a_row[k]);
                __m256 w_vec = _mm256_loadu_ps(&w[k * DimensionNew + j]);
                acc = _mm256_fmadd_ps(a_vec, w_vec, acc);
            }

            // Reduce SIMD accumulator
            float acc_buf[8];
            _mm256_storeu_ps(acc_buf, acc);
            for (int t = 0; t < 8; t++) sum += acc_buf[t];

            // Handle leftovers
            for (; k < DimensionOld; k++) sum += a_row[k] * w[k * DimensionNew + j];

            o_row[j] = sum;
        }
    } 
    // =========================================== ADJUST ===========================================   
}