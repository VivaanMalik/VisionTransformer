#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "include/ViT.h"

int main() {
    char *Labels = malloc(sizeof(char) * DATASET_SIZE);
    Tensor4 Images = LoadCIFAR10Dataset("dataset/cifar-10-batches-bin/train_all.bin", Labels, 0);
    Tensor4 ResizedImages = ResizeTo224(Images);
    free_tensor4(Images);
    Tensor3 PatchedImages = MakePatches(ResizedImages);
}