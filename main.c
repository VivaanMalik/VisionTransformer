#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "include/ViT.h"

int main() {
    time_t start_time = time(NULL);
    int num_b = DATASET_SIZE/DATASET_BATCH_SIZE;

    // Define all the Learnable Tensors
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

    Tensor PositionEncode = {0};
    PositionEncode.ndim = 2;
    int PEshape[2] = {NUM_PATCHES*NUM_PATCHES, PROJECTION_SIZE};
    PositionEncode.shape = PEshape;
    randomWeights(&PositionEncode);

    Tensor Q_W = {0};
    Q_W.ndim = 2;
    int QWshape[2] = {PROJECTION_SIZE, NUM_HEAD*HEAD_SIZE};
    Q_W.shape=QWshape;
    randomWeights(&Q_W);

    Tensor K_W = {0};
    K_W.ndim = 2;
    int KWshape[2] = {PROJECTION_SIZE, NUM_HEAD*HEAD_SIZE};
    K_W.shape=KWshape;
    randomWeights(&K_W);
    
    Tensor V_W = {0};
    V_W.ndim = 2;
    int VWshape[2] = {PROJECTION_SIZE, NUM_HEAD*HEAD_SIZE};
    V_W.shape=VWshape;
    randomWeights(&V_W);

    Tensor MLPHiddenWeights;
    MLPHiddenWeights.ndim = 2;
    int MLPHWshape[2] = {PROJECTION_SIZE, MLP_PROJECTION_SIZE};
    MLPHiddenWeights.shape = MLPHWshape;
    randomWeights(&MLPHiddenWeights);
    // MLPHiddenWeights.data = malloc(sizeof(float) * PROJECTION_SIZE * MLP_PROJECTION_SIZE);

    Tensor MLPHiddenBiases;
    MLPHiddenBiases.ndim = 1;
    int MLPHBshape[1] = {MLP_PROJECTION_SIZE};
    MLPHiddenBiases.shape = MLPHBshape;
    MLPHiddenBiases.data = calloc(MLP_PROJECTION_SIZE, sizeof(float));
    
    Tensor MLPOutputWeights;
    MLPOutputWeights.ndim = 2;
    int MLPOWshape[2] = {MLP_PROJECTION_SIZE, PROJECTION_SIZE};
    MLPOutputWeights.shape = MLPOWshape;
    // MLPOutputWeights.data = malloc(sizeof(float) * PROJECTION_SIZE * MLP_PROJECTION_SIZE);
    randomWeights(&MLPOutputWeights);

    Tensor MLPOutputBiases;
    MLPOutputBiases.ndim = 1;
    int MLPOBshape[1] = {PROJECTION_SIZE};
    MLPOutputBiases.shape = MLPOBshape;
    MLPOutputBiases.data = calloc(PROJECTION_SIZE, sizeof(float));

    Tensor AttentionProjectionWeights = {0};
    AttentionProjectionWeights.ndim = 2;
    int APWshape[2] = {NUM_HEAD*HEAD_SIZE, PROJECTION_SIZE};
    AttentionProjectionWeights.shape=APWshape;
    randomWeights(&AttentionProjectionWeights);

    Tensor Gamma;
    Gamma.ndim = 1;
    int Gshape[1] = {PROJECTION_SIZE};
    Gamma.shape = Gshape;
    Gamma.data = malloc(sizeof(float) * PROJECTION_SIZE);
    memset(Gamma.data, 1, sizeof(float) * PROJECTION_SIZE);

    Tensor Beta;
    Beta.ndim = 1;
    int Bshape[1] = {PROJECTION_SIZE};
    Beta.shape = Bshape;
    Beta.data = calloc(PROJECTION_SIZE, sizeof(float));

    for (int Batch = 0; Batch < num_b; Batch++) {

        // variables ill keep on using
        int ImageOffset, PatchOffset, HeadOffset, ImageOffset2, HeadOffset2, QiOffset, QiOffset2, PatchOffset2;


        // if ((Batch+1)%125==0 || Batch==num_b-1)
            printf("Batch: \t%d/%d\t%.2f%%\t%lds\n", Batch+1, num_b, (float)100*(Batch+1)/num_b, time(NULL)-start_time);

        Tensor Images = {0};           // Images -> 16 Image -> 64 patches (4x4) -> 3 color channels -> 16 pixels
        char Labels[DATASET_BATCH_SIZE];
        LoadCIFAR10Dataset("dataset/cifar-10-batches-bin/train_all.bin", &Images, Labels, Batch);

        // Projection
        Tensor EmbeddedImages = {0};
        EmbeddedImages.ndim = 3;
        int EIshape[3] = {DATASET_BATCH_SIZE, NUM_PATCHES*NUM_PATCHES, PROJECTION_SIZE};
        EmbeddedImages.shape = EIshape;
        float *EIdata = malloc(sizeof(float) * DATASET_BATCH_SIZE*NUM_PATCHES*NUM_PATCHES*PROJECTION_SIZE);
        EmbeddedImages.data = EIdata;

        for (int Image = 0; Image < DATASET_BATCH_SIZE; Image++) {
            ImageOffset = (NUM_PATCHES*NUM_PATCHES*PROJECTION_SIZE*Image);
            for (int p = 0; p < NUM_PATCHES*NUM_PATCHES; p++) {
                PatchOffset = PROJECTION_SIZE*p;
                for (int project = 0; project < PROJECTION_SIZE; project++) {
                    float sum = ProjectionBiases.data[project];
                    for (int pxl = 0; pxl < 3*PATCH_SIZE*PATCH_SIZE; pxl++) {
                        sum+=Images.data[3*IMAGE_SIZE*IMAGE_SIZE*Image + 3*PATCH_SIZE*PATCH_SIZE*p + pxl] * ProjectionWeights.data[3*project*PATCH_SIZE*PATCH_SIZE + pxl];
                    }
                    EmbeddedImages.data[ImageOffset + PatchOffset + project] = sum;
                    // if ((NUM_PATCHES*NUM_PATCHES*PROJECTION_SIZE*Image) + (PROJECTION_SIZE*p) + project > 409000000)
                    //     printf("%d ", (NUM_PATCHES*NUM_PATCHES*PROJECTION_SIZE*Image) + (PROJECTION_SIZE*p) + project);
                }
            }
        }
        freeTensor(&Images);

        // Positional Encoding, we add patch size tensor to each patch
        for (int Image = 0; Image < DATASET_BATCH_SIZE; Image++) {
            for (int i = 0; i < NUM_PATCHES*NUM_PATCHES*PROJECTION_SIZE; i++) {
                EmbeddedImages.data[Image*NUM_PATCHES*NUM_PATCHES*PROJECTION_SIZE + i] += PositionEncode.data[i];
            }
        }

        // Multihead Key Value Queries
        Tensor Q={0}, K={0}, V={0};
        Q.ndim = 4;
        int Qshape[4] = {DATASET_BATCH_SIZE, NUM_PATCHES*NUM_PATCHES, NUM_HEAD, HEAD_SIZE};
        Q.shape = Qshape;
        float *Qdata = malloc(sizeof(float) * DATASET_BATCH_SIZE * NUM_PATCHES*NUM_PATCHES *NUM_HEAD * HEAD_SIZE);
        Q.data = Qdata;

        K.ndim = 4;
        int Kshape[4] = {DATASET_BATCH_SIZE, NUM_PATCHES*NUM_PATCHES, NUM_HEAD, HEAD_SIZE};
        K.shape = Kshape;
        float *Kdata = malloc(sizeof(float) * DATASET_BATCH_SIZE * NUM_PATCHES*NUM_PATCHES *NUM_HEAD * HEAD_SIZE);
        K.data = Kdata;

        V.ndim = 4;    
        int Vshape[4] = {DATASET_BATCH_SIZE, NUM_PATCHES*NUM_PATCHES, NUM_HEAD, HEAD_SIZE};
        V.shape = Vshape;
        float *Vdata = malloc(sizeof(float) * DATASET_BATCH_SIZE * NUM_PATCHES*NUM_PATCHES *NUM_HEAD * HEAD_SIZE);
        V.data = Vdata;

        for (int Image = 0; Image < DATASET_BATCH_SIZE; Image++) {
            ImageOffset = Image*NUM_PATCHES*NUM_PATCHES*NUM_HEAD*HEAD_SIZE;
            for (int p = 0; p < NUM_PATCHES*NUM_PATCHES; p++) {
                PatchOffset = p*NUM_HEAD*HEAD_SIZE;
                for (int p2 = 0; p2 < NUM_HEAD*HEAD_SIZE; p2++) {
                    float q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;
                    for (int p3 = 0; p3 < PROJECTION_SIZE; p3++) {
                        float x = EmbeddedImages.data[Image*NUM_PATCHES*NUM_PATCHES*PROJECTION_SIZE + p*PROJECTION_SIZE + p3];
                        q_sum += x * Q_W.data[p3*NUM_HEAD*HEAD_SIZE+p2];
                        k_sum += x * K_W.data[p3*NUM_HEAD*HEAD_SIZE+p2];
                        v_sum += x * V_W.data[p3*NUM_HEAD*HEAD_SIZE+p2];
                    }
                    Q.data[ImageOffset + PatchOffset + p2] = q_sum;
                    K.data[ImageOffset + PatchOffset + p2] = k_sum;
                    V.data[ImageOffset + PatchOffset + p2] = v_sum;
                }
            }
        }

        // reinterpretation (values may vary)
        // Q, K, V -> Batch of 16 Queries, Keys, Values
        // Each Query, Key, Value -> 64 Patches
        // Each Patch -> 4 Heads
        // Head -> 32 values  

        // Scores -> (element of batch)wise -> patch -> head -> other patch relation
        Tensor Scores = {0};
        Scores.ndim = 4;
        int Sshape[4] = {DATASET_BATCH_SIZE, NUM_PATCHES*NUM_PATCHES, NUM_HEAD, NUM_PATCHES*NUM_PATCHES};
        Scores.shape = Sshape;
        float *Sdata = malloc(sizeof(float) * DATASET_BATCH_SIZE * NUM_PATCHES*NUM_PATCHES *NUM_HEAD * NUM_PATCHES*NUM_PATCHES);
        Scores.data = Sdata;

        for (int Image = 0; Image < DATASET_BATCH_SIZE; Image++) {
            ImageOffset = NUM_PATCHES*NUM_PATCHES*NUM_HEAD*HEAD_SIZE*Image;
            ImageOffset2 = NUM_PATCHES*NUM_PATCHES*NUM_HEAD*NUM_PATCHES*NUM_PATCHES*Image;
            for (int qi = 0; qi < NUM_PATCHES*NUM_PATCHES; qi++) {
                QiOffset = qi*(NUM_HEAD*NUM_PATCHES*NUM_PATCHES);
                for (int h = 0; h < NUM_HEAD; h++) {
                    HeadOffset = h*HEAD_SIZE;
                    HeadOffset2 = h*NUM_PATCHES*NUM_PATCHES;
                    for (int ki = 0; ki < NUM_PATCHES*NUM_PATCHES; ki++) {
                        float sum = .0f;
                        for (int pxl = 0; pxl < HEAD_SIZE; pxl++) {
                            int indx = ImageOffset + HeadOffset + pxl;
                            sum += Q.data[indx + qi*NUM_HEAD*HEAD_SIZE] * K.data[indx + ki*NUM_HEAD*HEAD_SIZE];
                        }
                        Scores.data[ImageOffset2 + QiOffset + HeadOffset2 + ki] = sum/sqrtf(HEAD_SIZE);
                    }
                }
            }
        }

        // add softmax to scores ka ki -> key
        // for softmax, 
        // for #1: get max(x)
        // #2: find and assign e^(x-max(x)), also sum it all up
        // #3: divide by sum -> probabilities

        Tensor AttentionWeights = {0};
        AttentionWeights.ndim = 4;
        int AWshape[4] = {DATASET_BATCH_SIZE, NUM_PATCHES*NUM_PATCHES, NUM_HEAD, NUM_PATCHES*NUM_PATCHES};
        AttentionWeights.shape = AWshape;
        float *AWdata = malloc(sizeof(float) * DATASET_BATCH_SIZE * NUM_PATCHES*NUM_PATCHES *NUM_HEAD * NUM_PATCHES*NUM_PATCHES);
        AttentionWeights.data = AWdata;

        for (int Image = 0; Image < DATASET_BATCH_SIZE; Image++) {
            ImageOffset = Image*NUM_PATCHES*NUM_PATCHES*NUM_HEAD*NUM_PATCHES*NUM_PATCHES;
            for (int qi = 0; qi < NUM_PATCHES*NUM_PATCHES; qi++) {
                QiOffset = NUM_HEAD*NUM_PATCHES*NUM_PATCHES*qi;
                for (int h = 0; h < NUM_HEAD; h++) {
                    HeadOffset = h*NUM_PATCHES*NUM_PATCHES;

                    // find max(x)
                    float max = -INFINITY;
                    int base = ImageOffset + QiOffset + HeadOffset;
                    for (int ki = 0; ki < NUM_PATCHES*NUM_PATCHES; ki++) {
                        float val = Scores.data[base + ki];
                        max = (max>val) ? max : val;
                    }

                    float sum = 0.0f;
                    for (int ki = 0; ki < NUM_PATCHES*NUM_PATCHES; ki++) {
                        float val = expf(Scores.data[base + ki] - max);
                        AttentionWeights.data[base + ki] = val;
                        sum+=val;
                    }

                    for (int ki = 0; ki < NUM_PATCHES*NUM_PATCHES; ki++) {
                        AttentionWeights.data[base + ki]/=sum;
                    }
                }
            }
        }
        
        // now we do weighted avg
        // Query wala 64 and Key wala 64 from AW
        // 64 for key and 32 for headsize from V
        // [64(Q), 64(K)][64(K), 32(H)]
        // AW[axis 1, 3]V[axis 1, 3]

        // Image -> qi -> h -> headsize -> ki

        Tensor AttentionOutput = {0};
        AttentionOutput.ndim = 3;
        int AOshape[3] = {DATASET_BATCH_SIZE, NUM_PATCHES*NUM_PATCHES, NUM_HEAD*HEAD_SIZE};
        AttentionOutput.shape = AOshape;
        float *AOdata = malloc(sizeof(float) * DATASET_BATCH_SIZE*NUM_PATCHES*NUM_PATCHES*NUM_HEAD*HEAD_SIZE);
        AttentionOutput.data = AOdata;

        for (int Image = 0; Image < DATASET_BATCH_SIZE; Image++) {
            ImageOffset = Image*NUM_PATCHES*NUM_PATCHES*NUM_HEAD*NUM_PATCHES*NUM_PATCHES;
            ImageOffset2 = Image*NUM_PATCHES*NUM_PATCHES*NUM_HEAD*HEAD_SIZE;
            for (int qi = 0; qi < NUM_PATCHES*NUM_PATCHES; qi++) {
                QiOffset = NUM_HEAD*NUM_PATCHES*NUM_PATCHES*qi;
                QiOffset2 = NUM_HEAD*HEAD_SIZE*qi;
                for (int h = 0; h < NUM_HEAD; h++) {
                    HeadOffset = h*NUM_PATCHES*NUM_PATCHES;
                    HeadOffset2 = h*HEAD_SIZE;
                    for (int pxl = 0; pxl < HEAD_SIZE; pxl++) {
                        float sum = 0;
                        for (int ki = 0; ki < NUM_PATCHES*NUM_PATCHES; ki++) {
                            sum+=AttentionWeights.data[ImageOffset + QiOffset + HeadOffset + ki] * V.data[ImageOffset2 + ki*NUM_HEAD*HEAD_SIZE + HeadOffset2 + pxl];
                        }
                        AttentionOutput.data[ImageOffset2 + QiOffset2 + HeadOffset2 + pxl] = sum;
                    }
                }
            }
        }      
        
        // now we reproject to PROJECTION_SIZE from NUM_HEAD * HEAD_SIZE and do residual
        for (int Image = 0; Image < DATASET_BATCH_SIZE; Image++) {
            ImageOffset = Image*NUM_PATCHES*NUM_PATCHES*PROJECTION_SIZE;
            ImageOffset2 = Image*NUM_PATCHES*NUM_PATCHES*NUM_HEAD*HEAD_SIZE;
            for (int p = 0; p < NUM_PATCHES*NUM_PATCHES; p++) {
                PatchOffset = p*PROJECTION_SIZE;
                PatchOffset2 = p*NUM_HEAD*HEAD_SIZE;
                for (int p2 = 0; p2 < PROJECTION_SIZE; p2++) {
                    float sum = 0.0f;
                    for (int pxl = 0; pxl < NUM_HEAD*HEAD_SIZE; pxl++) {
                        sum+=AttentionProjectionWeights.data[pxl*PROJECTION_SIZE + p2] * AttentionOutput.data[ImageOffset2 + PatchOffset2 + pxl];
                    }
                    EmbeddedImages.data[ImageOffset + PatchOffset + p2] += sum;
                }
            }
        }

        // we do layernormmmmmm
        // welford to find mean and variance
        Tensor mean, variance;
        mean.ndim = 2;
        variance.ndim = 2;
        int mshape[2] = {DATASET_BATCH_SIZE, NUM_PATCHES*NUM_PATCHES};
        int vshape[2] = {DATASET_BATCH_SIZE, NUM_PATCHES*NUM_PATCHES};
        mean.shape = mshape;
        variance.shape = vshape;
        mean.data = malloc(sizeof(float) * DATASET_BATCH_SIZE * NUM_PATCHES*NUM_PATCHES);
        variance.data = malloc(sizeof(float) * DATASET_BATCH_SIZE * NUM_PATCHES*NUM_PATCHES);

        for (int Image = 0; Image < DATASET_BATCH_SIZE; Image++) {
            ImageOffset = Image*NUM_PATCHES*NUM_PATCHES;
            ImageOffset2 = ImageOffset*PROJECTION_SIZE;
            for (int p = 0; p < NUM_PATCHES*NUM_PATCHES; p++) {
                PatchOffset = p*PROJECTION_SIZE;
                mean.data[ImageOffset + p] = 0;
                variance.data[ImageOffset + p] = 0;
                for (int pxl = 0; pxl < PROJECTION_SIZE; pxl++) {
                    float x = EmbeddedImages.data[ImageOffset2 + PatchOffset + pxl];
                    float delta = x - mean.data[ImageOffset + p];
                    mean.data[ImageOffset + p] += delta / (pxl +1);
                    float delta2 = x - mean.data[ImageOffset + p];
                    variance.data[ImageOffset + p] += delta * delta2;
                }
                variance.data[ImageOffset + p] /= PROJECTION_SIZE;

                for (int pxl = 0; pxl < PROJECTION_SIZE; pxl++) {
                    float x = EmbeddedImages.data[ImageOffset2 + PatchOffset + pxl];
                    float normed = (x - mean.data[ImageOffset + p]) / sqrtf(variance.data[ImageOffset + p] + 1e-5f);
                    EmbeddedImages.data[ImageOffset2 + PatchOffset + pxl] = normed * Gamma.data[pxl] + Beta.data[pxl];
                }
            }
        }

        Tensor MLPHidden;
        MLPHidden.ndim = 3;
        int MLPHshape[3] = {DATASET_BATCH_SIZE, NUM_PATCHES*NUM_PATCHES, MLP_PROJECTION_SIZE};
        MLPHidden.shape = MLPHshape;
        MLPHidden.data = malloc(sizeof(float) * DATASET_BATCH_SIZE*NUM_PATCHES*NUM_PATCHES*MLP_PROJECTION_SIZE);

        Tensor MLPOutput;
        MLPOutput.ndim = 3;
        int MLPOshape[3] = {DATASET_BATCH_SIZE, NUM_PATCHES*NUM_PATCHES, PROJECTION_SIZE};
        MLPOutput.shape = MLPOshape;
        MLPOutput.data = malloc(sizeof(float) * DATASET_BATCH_SIZE*NUM_PATCHES*NUM_PATCHES*PROJECTION_SIZE);
        
        for (int Image = 0; Image < DATASET_BATCH_SIZE; Image++) {
            ImageOffset = Image*NUM_PATCHES*NUM_PATCHES*PROJECTION_SIZE;
            ImageOffset2 = Image*NUM_PATCHES*NUM_PATCHES*MLP_PROJECTION_SIZE;
            for (int p = 0; p < NUM_PATCHES*NUM_PATCHES; p++) {
                PatchOffset = p*PROJECTION_SIZE;
                PatchOffset2 = p*MLP_PROJECTION_SIZE;
                for (int p2 = 0; p2 < MLP_PROJECTION_SIZE; p2++) {
                    float sum = 0;
                    for (int pxl = 0; pxl < PROJECTION_SIZE; pxl++) {
                        sum+=MLPHiddenWeights.data[pxl*MLP_PROJECTION_SIZE + p2] * EmbeddedImages.data[ImageOffset + PatchOffset + pxl];
                    }
                    MLPHidden.data[ImageOffset2 + PatchOffset2 + p2] = sum + MLPHiddenBiases.data[p2];

                    // GELU approximation using tanh
                    float x = MLPHidden.data[ImageOffset2 + PatchOffset2 + p2];
                    MLPHidden.data[ImageOffset2 + PatchOffset2 + p2] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
                }
            }
        }



        for (int Image = 0; Image < DATASET_BATCH_SIZE; Image++) {
            ImageOffset = Image*NUM_PATCHES*NUM_PATCHES*PROJECTION_SIZE;
            ImageOffset2 = Image*NUM_PATCHES*NUM_PATCHES*MLP_PROJECTION_SIZE;
            for (int p = 0; p < NUM_PATCHES*NUM_PATCHES; p++) {
                PatchOffset = p*PROJECTION_SIZE;
                PatchOffset2 = p*MLP_PROJECTION_SIZE;
                for (int pxl = 0; pxl < PROJECTION_SIZE; pxl++) {
                    float sum = 0;
                    for (int p2 = 0; p2 < MLP_PROJECTION_SIZE; p2++) {
                        sum+=MLPOutputWeights.data[p2*PROJECTION_SIZE + pxl] * MLPHidden.data[ImageOffset2 + PatchOffset2 + p2];
                    }
                    MLPOutput.data[ImageOffset + PatchOffset + pxl] = sum + MLPOutputBiases.data[pxl];
                    EmbeddedImages.data[ImageOffset + PatchOffset + pxl] += MLPOutput.data[ImageOffset + PatchOffset + pxl];
                }
            }
        }

        for (int Image = 0; Image < DATASET_BATCH_SIZE; Image++) {
            ImageOffset = Image*NUM_PATCHES*NUM_PATCHES;
            ImageOffset2 = ImageOffset*PROJECTION_SIZE;
            for (int p = 0; p < NUM_PATCHES*NUM_PATCHES; p++) {
                PatchOffset = p*PROJECTION_SIZE;
                mean.data[ImageOffset + p] = 0;
                variance.data[ImageOffset + p] = 0;
                for (int pxl = 0; pxl < PROJECTION_SIZE; pxl++) {
                    float x = EmbeddedImages.data[ImageOffset2 + PatchOffset + pxl];
                    float delta = x - mean.data[ImageOffset + p];
                    mean.data[ImageOffset + p] += delta / (pxl +1);
                    float delta2 = x - mean.data[ImageOffset + p];
                    variance.data[ImageOffset + p] += delta * delta2;
                }
                variance.data[ImageOffset + p] /= PROJECTION_SIZE;

                for (int pxl = 0; pxl < PROJECTION_SIZE; pxl++) {
                    float x = EmbeddedImages.data[ImageOffset2 + PatchOffset + pxl];
                    float normed = (x - mean.data[ImageOffset + p]) / sqrtf(variance.data[ImageOffset + p] + 1e-5f);
                    EmbeddedImages.data[ImageOffset2 + PatchOffset + pxl] = normed * Gamma.data[pxl] + Beta.data[pxl];
                }
            }
        }

        // free shit cuz otherwise memleak
        freeTensorData(&EmbeddedImages);
        freeTensorData(&AttentionOutput);
        freeTensorData(&mean);
        freeTensorData(&variance);
        freeTensorData(&MLPHidden);
        freeTensorData(&MLPOutput);
        freeTensorData(&Scores);
        freeTensorData(&AttentionWeights);
        freeTensorData(&Q);
        freeTensorData(&K);
        freeTensorData(&V);
    }
    freeTensorData(&Gamma);
    freeTensorData(&Beta);
    freeTensorData(&ProjectionWeights);
    freeTensorData(&ProjectionWeights);
    freeTensorData(&PositionEncode);
    freeTensorData(&AttentionProjectionWeights);
    freeTensorData(&MLPHiddenWeights);
    freeTensorData(&MLPHiddenBiases);
    freeTensorData(&MLPOutputWeights);
    freeTensorData(&MLPOutputBiases);
    freeTensorData(&Q_W);
    freeTensorData(&K_W);
    freeTensorData(&V_W);
}