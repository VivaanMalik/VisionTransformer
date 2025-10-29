#include "../include/image.h"
#include "../include/ViT.h"
#include <stdlib.h>
#include <stdio.h>

Tensor4 LoadCIFAR10Dataset(const char *Path, char *Labels, int Batch) {
    FILE *f = fopen(Path, "rb");
    if (!f) { printf("File not found\n"); exit(0); }

    fseek(f, Batch*DATASET_BATCH_SIZE*(IMAGE_SIZE*IMAGE_SIZE*3+1), SEEK_CUR);

    Tensor4 Images = alloc_tensor4(DATASET_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE);

    unsigned char l;
    unsigned char buffer[IMAGE_SIZE*IMAGE_SIZE*3];
    for (int b = 0; b < DATASET_BATCH_SIZE; b++) {
        if (fread(&l, 1, 1, f) != 1) {
            printf("End of file or read error\n");
            exit(0);
        }
        Labels[b] = l;

        if (fread(buffer, 1, IMAGE_SIZE*IMAGE_SIZE*3, f) != IMAGE_SIZE*IMAGE_SIZE*3) {
            printf("End of file or read error\n");
            exit(0);
        }
        for (int rgb = 0; rgb < 3; rgb++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                for (int y = 0; y < IMAGE_SIZE; y++) {
                    T4(Images, b, rgb, x, y) = buffer[rgb*IMAGE_SIZE*IMAGE_SIZE + x*IMAGE_SIZE + y]/255.0f;
                }
            }
        }
    }
    return Images;
}

Tensor4 ResizeTo224(Tensor4 input) {

    Tensor4 output = alloc_tensor4(DATASET_BATCH_SIZE, 3, IMAGE_SCALING, IMAGE_SCALING);
    const float scale = (float)((IMAGE_SIZE-1)/(IMAGE_SCALING-1));

    for (int b=0; b<DATASET_BATCH_SIZE; b++)
        for (int c=0; c<3; c++)
            for (int x=0; x<IMAGE_SCALING; x++) {

                float x_in = x * scale;
                int x0 = (int)x_in;
                int x1 = (x0 < IMAGE_SIZE - 1) ? x0 + 1 : x0;
                float dx = x_in - x0;

                for (int y=0; y<IMAGE_SCALING; y++) {
                    float y_in = y * scale;
                    int y0 = (int)y_in;
                    int y1 = (y0 < IMAGE_SIZE - 1) ? y0 + 1 : y0;
                    float dy = y_in - y0;

                    float p00 = T4(input, b, c, x0, y0);
                    float p01 = T4(input, b, c, x0, y1);
                    float p10 = T4(input, b, c, x1, y0);
                    float p11 = T4(input, b, c, x1, y1);

                    float v0 = p00 * (1 - dy) + p01 * dy;
                    float v1 = p10 * (1 - dy) + p11 * dy;
                    float val = v0 * (1 - dx) + v1 * dx;

                    T4(output, b, c, x, y) = val;
                }
            }
    return output;
}

Tensor4 MakePatches(Tensor4 input) {

    int NumPatches = NUM_PATCHES * NUM_PATCHES;
    Tensor4 output = alloc_tensor4(DATASET_BATCH_SIZE, NumPatches, 3, PATCH_SIZE*PATCH_SIZE);

    for (int b = 0; b < DATASET_BATCH_SIZE; b++) {
        int p = 0;
        for (int py = 0; py < IMAGE_SCALING; py += PATCH_SIZE) {
            for (int px = 0; px < IMAGE_SCALING; px += PATCH_SIZE) {

                for (int c = 0; c < 3; c++) {
                    for (int dy = 0; dy < PATCH_SIZE; dy++) {
                        for (int dx = 0; dx < PATCH_SIZE; dx++) {

                            int flat = dy * PATCH_SIZE + dx;
                            T4(output, b, p, c, flat) = T4(input, b, c, py + dy, px + dx);
                        }
                    }
                }
                p++;
            }
        }
    }

    return output;
}