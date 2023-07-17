#ifndef _EOS_TENSOR_H
#define _EOS_TENSOR_H
#include "stdlib.h"

#define TENSOR_IDX(tensor, row, col, channel) (tensor.dataoffset + (row) * tensor.row_stride + (col) * tensor.col_stride + (channel) * tensor.channel_stride)
#define TENSOR_AT(tensor, row, col, channel) (tensor.data[TENSOR_IDX(tensor, row, col, channel)])

typedef struct Eos_Tensor3f
{
    size_t rows;
    size_t cols;
    size_t channels;
    size_t dataoffset;
    size_t row_stride; // size to next row
    size_t col_stride; // size to next col
    size_t channel_stride;
    float *data;
} Eos_Tensor3f;

Eos_Tensor3f eos_tensor3f_alloc(size_t rows, size_t cols, size_t channels); // The only place where memory is heap-allocated, all else are shallow copies.
Eos_Tensor3f eos_tensor3f_borrow(size_t rows, size_t cols, size_t channels, float *data);

void eos_tensor3f_free(Eos_Tensor3f *tensor);

void eos_tensor3f_print(Eos_Tensor3f tensor);
void eos_tensor3f_info(Eos_Tensor3f tensor);

void eos_tensor3f_random(Eos_Tensor3f tensor, float min, float max);
void eos_tensor3f_zero(Eos_Tensor3f tensor);

#endif