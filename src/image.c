#include "../include/image.h"
#include "../include/tensor.h"
#include <stdlib.h>
#include <stdio.h>

void LoadCIFAR10Dataset(const char *Path, Tensor *Images, char *Labels) {
    FILE *f = fopen(Path, "rb");
    if (!f) { printf("File not found\n"); return; }

    Images->ndim = 4;
    Images->shape = malloc(sizeof(int)*4);
    Images->shape[0] = DATASET_BATCH_SIZE;
    Images->shape[1] = NUM_PATCHES*NUM_PATCHES;
    Images->shape[2] = 3;
    Images->shape[3] = PATCH_SIZE*PATCH_SIZE;
    Images->data = malloc(sizeof(float) * DATASET_BATCH_SIZE*IMAGE_SIZE*IMAGE_SIZE*3);

    unsigned char l;
    unsigned char buffer[IMAGE_SIZE*IMAGE_SIZE*3];
    for (int i = 0; i < DATASET_BATCH_SIZE; i++) {
        if (fread(&l, 1, 1, f) != 1) {
            printf("End of file or read error\n");
            return;
        }
        Labels[i] = l;

        if (fread(buffer, 1, IMAGE_SIZE*IMAGE_SIZE*3, f) != IMAGE_SIZE*IMAGE_SIZE*3) {
            printf("End of file or read error\n");
            return;
        }
        for (int y = 0; y < NUM_PATCHES; y++) {
            for (int x = 0; x < NUM_PATCHES; x++) {
                for (int ii = 0; ii < PATCH_SIZE; ii++) {
                    for (int jj = 0; jj < PATCH_SIZE; jj++) {
                        for (int rgb = 0; rgb < 3; rgb++) {
                            Images->data[(IMAGE_SIZE*IMAGE_SIZE*3*i) + ((3*(y*NUM_PATCHES+x)+rgb)*PATCH_SIZE*PATCH_SIZE) + ii*PATCH_SIZE + jj] = buffer[rgb*IMAGE_SIZE*IMAGE_SIZE + y*IMAGE_SIZE*PATCH_SIZE + x*PATCH_SIZE + IMAGE_SIZE*ii + jj]/255.0f;
                        }
                    }
                }
            }
        }
    }
}