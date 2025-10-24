#include <stdio.h>
#include "include/ViT.h"

int main() {
    print(HELLO_WORLD);

    Tensor A;
    int shape1[2] = {2, 2};
    A.shape=shape1;
    float arr1[4] = {1, 2, 3, 4};
    A.data = arr1;
    A.ndim=2;

    Tensor B;
    int shape2[2] = {2, 2};
    B.shape=shape2;
    float arr2[4] = {4, 3, 2, 1};
    B.data = arr2;
    B.ndim=2;

    printTensor(&A);
    printf("\t\tA\n\n");

    printTensor(&B);
    printf("\t\tB\n\n");

    Tensor C = {0}; // make sure value hai
    addTensor(&A, &B, &C);
    printTensor(&C);
    printf("\t\tC\n\n");
    
    // freeTensor(&C);
    matmulTensor(&A, &B, &C);
    printTensor(&C);
    printf("\t\tC\n\n");

    scaleTensor(&C, 0.1f);
    printTensor(&C);
    printf("\t\tC\n\n");

    Tensor C_T = {0}; // make sure value hai
    transposeTensor(&C, &C_T);
    printTensor(&C_T);
    printf("\t\tC_T\n\n");
    freeTensor(&C_T);

    softmaxTensor(&C);
    printTensor(&C);
    printf("\t\tC\n\n");
    // printTensor(&C_T);           // shpuld segfault
    // printf("\n\n");

    copyTensor(&A, &C);
    printTensor(&C);
    printf("\t\tC\n\n");

    float arr[2] = {1, -1};
    addBiasToTensor(&C, arr);
    printTensor(&C);
    printf("\t\tC\n\n");

    // freeTensor(&A);
    // freeTensor(&B);
    // freeTensor(&C);

    Tensor Images;           // Images -> Image -> 64 patches (4x4) -> 3 color channels
    char Labels[50000];
    LoadCIFAR10Dataset("dataset/cifar-10-batches-bin/train_all.bin", &Images, Labels);

 
    printf("\n");
}