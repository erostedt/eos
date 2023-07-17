#include "stdio.h"
#include "stdlib.h"
#include "assert.h"
#include "tensor.h"

#define TENSOR_NPIXELS(tensor) ((tensor).rows * (tensor).cols)
#define TENSOR_NBYTES(tensor) ((tensor).rows * (tensor).cols * (tensor).channels * sizeof(float))
#define NOT_IMPLEMENTED do {printf(__func__);\
                        printf(" is not implemented.\n");\
                        assert(0);\
                       } while(0)


Eos_Tensor3f eos_tensor3f_alloc(size_t rows, size_t cols, size_t channels)
{
    assert(rows > 0 && cols > 0 && channels > 0);
    float *data = (float *)malloc(rows * cols *  channels * sizeof(float));
    assert(data != NULL);
    return eos_tensor3f_borrow(rows, cols, channels, data);
}

Eos_Tensor3f eos_tensor3f_borrow(size_t rows, size_t cols, size_t channels, float *data)
{
    assert(rows > 0 && cols > 0 && channels > 0);
    return (Eos_Tensor3f) {
        .rows = rows,
        .cols = cols,
        .channels = channels,
        .row_stride = cols * channels,
        .col_stride = channels,
        .channel_stride = 1,
        .dataoffset = 0,
        .data = data    
    };
}

void eos_tensor3f_free(Eos_Tensor3f *tensor)
{
    tensor->rows = tensor->cols = tensor->channels = tensor->row_stride = tensor->col_stride = tensor->channel_stride = tensor->dataoffset = 0;
    free(tensor->data);
    tensor->data = NULL;
    tensor = NULL;
}

void eos_tensor3f_print(Eos_Tensor3f tensor)
{
    printf("[\n");
    for (size_t row = 0; row < tensor.rows; row++)
    {
        printf("  [\n");
        for (size_t col = 0; col < tensor.cols-1; col++)
        {
            printf("    [");
            for (size_t channel = 0; channel < tensor.channels-1; channel++)
            {
                printf("%f, ", TENSOR_AT(tensor, row, col, channel));
            }
            printf("%f] \n", TENSOR_AT(tensor, row, col, tensor.channels-1));
        }

        printf("%f] \n", TENSOR_AT(tensor, row, tensor.cols-1, tensor.channels-1));
    }
    printf("]\n");

}

void eos_tensor3f_info(Eos_Tensor3f tensor)
{
    printf("Shape: (%zu, %zu, %zu)\n", tensor.rows, tensor.cols, tensor.channels);
    printf("Strides: (%zu, %zu, %zu)\n", tensor.row_stride, tensor.col_stride, tensor.channel_stride);
    printf("Num bytes: %zu\n", TENSOR_NBYTES(tensor));
    printf("Data offset: %zu\n", tensor.dataoffset);
}


void eos_tensor3f_random(Eos_Tensor3f tensor, float min, float max)
{
    assert(min < max);
    float unit_random;
    for (size_t row = 0; row < tensor.rows; row++)
    {
        for (size_t col = 0; col < tensor.cols; col++)
        {
            for (size_t channel = 0; channel < tensor.channels; channel++)
            {
                unit_random = (float)rand()/(float)RAND_MAX;
                TENSOR_AT(tensor, row, col, channel) = unit_random * (max - min) + min;
            }
        }
    }
}


void eos_tensor3f_fill(Eos_Tensor3f tensor, float value)
{
    for (size_t row = 0; row < tensor.rows; row++)
    {
        for (size_t col = 0; col < tensor.cols; col++)
        {
            for (size_t channel = 0; channel < tensor.channels; channel++)
                TENSOR_AT(tensor, row, col, channel) = value;
        }
    }
}

void eos_tensor3f_zero(Eos_Tensor3f tensor)
{
    eos_tensor3f_fill(tensor, 0.0f);
}

