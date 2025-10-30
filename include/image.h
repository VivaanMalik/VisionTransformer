#ifndef IMAGE_H
#define IMAGE_H

#include "structs.h"

Tensor4 LoadCIFAR10Dataset(const char *, char *, int);
Tensor4 ResizeTo224(Tensor4);
Tensor3 MakePatches(Tensor4);

#endif