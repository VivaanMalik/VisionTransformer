#ifndef ViT
#define ViT

#include "structs.h"
#include "image.h"

#define DATASET_SIZE 50000          // 50000
#define IMAGE_SIZE 32               // 32
#define PATCH_SIZE 16               // 4
#define IMAGE_SCALING 224
#define NUM_PATCHES (IMAGE_SCALING/PATCH_SIZE)

#define PROJECTION_SIZE 128
#define DATASET_BATCH_SIZE 16
#define NUM_HEAD 4
#define HEAD_SIZE 32
#define MLP_PROJECTION_SIZE 4*PROJECTION_SIZE

#endif