#ifndef IMAGE_H
#define IMAGE_H

#include "tensor.h"

#define DATASET_SIZE 10000          // 50000
#define IMAGE_SIZE 32               // 32
#define PATCH_SIZE 4                // 4
#define NUM_PATCHES (IMAGE_SIZE/PATCH_SIZE)

void LoadCIFAR10Dataset(const char *, Tensor *, char *, int);

#endif