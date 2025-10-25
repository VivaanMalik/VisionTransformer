#include <stdio.h>
#include <stdlib.h>
#include "include/ViT.h"

int main() {
    for (int Batch = 0; Batch < DATASET_SIZE/DATASET_BATCH_SIZE; Batch++) {
        Tensor Images = {0};           // Images -> Image -> 64 patches (4x4) -> 3 color channels
        char Labels[DATASET_BATCH_SIZE];
        LoadCIFAR10Dataset("dataset/cifar-10-batches-bin/train_all.bin", &Images, Labels, Batch);

        // Projection
        Tensor ProjectionWeights = {0};
        ProjectionWeights.ndim = 2;
        int PWshape[2] = {PROJECTION_SIZE, 3*PATCH_SIZE*PATCH_SIZE};
        ProjectionWeights.shape=PWshape;
        randomWeights(&ProjectionWeights);

        Tensor ProjectionBiases = {0};
        ProjectionBiases.ndim = 1;
        int PBshape[1] = {PROJECTION_SIZE};
        ProjectionBiases.shape=PBshape;
        float PBdata[PROJECTION_SIZE] = {0};
        ProjectionBiases.data = PBdata;

        Tensor EmbeddedImages = {0};
        EmbeddedImages.ndim = 3;
        int EIshape[3] = {DATASET_BATCH_SIZE, NUM_PATCHES*NUM_PATCHES, PROJECTION_SIZE};
        EmbeddedImages.shape = EIshape;
        float *EIdata = malloc(sizeof(float) * DATASET_BATCH_SIZE*NUM_PATCHES*NUM_PATCHES*PROJECTION_SIZE);
        EmbeddedImages.data = EIdata;
        for (int Image = 0; Image < DATASET_BATCH_SIZE; Image++) {
            for (int p = 0; p < NUM_PATCHES*NUM_PATCHES; p++) {
                for (int project = 0; project < PROJECTION_SIZE; project++) {
                    float sum = ProjectionBiases.data[project];
                    for (int pxl = 0; pxl < 3*PATCH_SIZE*PATCH_SIZE; pxl++) {
                        sum+=Images.data[3*IMAGE_SIZE*IMAGE_SIZE*Image + 3*PATCH_SIZE*PATCH_SIZE*p + pxl] * ProjectionWeights.data[3*project*PATCH_SIZE*PATCH_SIZE + pxl];
                    }
                    EmbeddedImages.data[(NUM_PATCHES*NUM_PATCHES*PROJECTION_SIZE*Image) + (PROJECTION_SIZE*p) + project] = sum;
                    // if ((NUM_PATCHES*NUM_PATCHES*PROJECTION_SIZE*Image) + (PROJECTION_SIZE*p) + project > 409000000)
                    //     printf("%d ", (NUM_PATCHES*NUM_PATCHES*PROJECTION_SIZE*Image) + (PROJECTION_SIZE*p) + project);
                }
            }
        }
        freeTensor(&Images);
    }
}