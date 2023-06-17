#ifndef _EOS_TENSOR_H
#define _EOS_TENSOR_H

#define TENSOR_IDX(tensor, row, col, channel) (tensor.dataoffset + (row) * tensor.row_stride + (col) * tensor.col_stride + (channel) * tensor.channel_stride)
#define TENSOR_AT(tensor, row, col, channel) (tensor.data[TENSOR_IDX(tensor, row, col, channel)])

typedef struct eos_tensor3f
{
    size_t rows;
    size_t cols;
    size_t channels;
    size_t dataoffset;
    size_t row_stride; // size to next row
    size_t col_stride; // size to next col
    size_t channel_stride;
    float *data;
} eos_tensor3f;

eos_tensor3f eos_tensor3f_alloc(size_t rows, size_t cols, size_t channels); // The only place where memory is heap-allocated, all else are shallow copies.
eos_tensor3f eos_tensor3f_borrow(size_t rows, size_t cols, size_t channels, float *data);

void eos_tensor3f_free(eos_tensor3f *tensor);

void eos_tensor3f_print(eos_tensor3f tensor);
void eos_tensor3f_info(eos_tensor3f tensor);

void eos_tensor3f_random(eos_tensor3f tensor, float min, float max);
void eos_tensor3f_zero(eos_tensor3f tensor);

void eos_tensor3f_cross_corr(eos_tensor3f dst, eos_tensor3f tensor, eos_tensor3f filter);

#endif