#ifndef TENSOR_H
#define TENSOR_H

typedef struct
{
    float *data;
    int *shape;
    int ndim;
} Tensor;

void addTensor(Tensor *, Tensor *, Tensor *);
void matmulTensor(Tensor *, Tensor *, Tensor *);
void scaleTensor(Tensor *, float);
void transposeTensor(Tensor *, Tensor *);
void softmaxTensor(Tensor *);
void addBiasToTensor(Tensor *, float *);
void randomWeights(Tensor *);

void copyTensor(Tensor *, Tensor *);
void freeTensor(Tensor *);
void freeTensorData(Tensor *);
void printTensor(Tensor *);
void project_last_axis(const Tensor * restrict A, const Tensor * restrict W, const Tensor * restrict Bias, Tensor * Out);


#endif