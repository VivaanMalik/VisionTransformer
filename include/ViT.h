#ifndef ViT
#define ViT

#include "structs.h"
#include "image.h"

#define DATASET_SIZE 50000          // 50000
#define IMAGE_SIZE 32               // 32
#define PATCH_SIZE 16               // 4
#define IMAGE_SCALING 224
#define NUM_HEAD 3

#define DATASET_BATCH_SIZE 1
#define NUM_BATCHES (DATASET_SIZE/DATASET_BATCH_SIZE)
#define NUM_TOKENS 196
#define TOKEN_SIZE 192
#define HEAD_SIZE 64



#endif