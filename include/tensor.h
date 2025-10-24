#ifndef TENSOR_H
#define TENSOR_H

typedef struct Tensor
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

void copyTensor(Tensor *, Tensor *);
void freeTensor(Tensor *);
void printTensor(Tensor *);


#endif