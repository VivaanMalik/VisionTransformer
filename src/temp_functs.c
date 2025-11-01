#include <math.h>
#include <float.h>  // for FLT_MAX
#include "struct.h"

// Compute row-wise stable softmax on Matrix input, store in output
void softmax_matrix(Matrix input, Matrix output) {
    if (input.rows != output.rows || input.cols != output.cols) {
        fprintf(stderr, "Error: softmax_matrix() dimension mismatch\n");
        return;
    }

    for (int r = 0; r < input.rows; r++) {
        // Step 1: find max in this row
        float max_val = -FLT_MAX;
        for (int c = 0; c < input.cols; c++) {
            if (M(input, r, c) > max_val)
                max_val = M(input, r, c);
        }

        // Step 2: compute exponentials of (x - max)
        float sum_exp = 0.0f;
        for (int c = 0; c < input.cols; c++) {
            float e = expf(M(input, r, c) - max_val);
            M(output, r, c) = e;
            sum_exp += e;
        }

        // Step 3: normalize by sum of exps
        for (int c = 0; c < input.cols; c++) {
            M(output, r, c) /= sum_exp;
        }
    }
}
