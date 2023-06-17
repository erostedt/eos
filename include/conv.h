#include "tensor.h"
#include "batch.h"

#ifndef _EOS_CONV_H
#define _EOS_CONV_H
typedef struct eos_conv_layer
{
    eos_batch4f batch;
    eos_batch4f filters;
    
    size_t filter_rows;
    size_t filter_cols;
    size_t filter_depth;

    eos_batch4f filter_gradients;
    eos_batch4f local_gradients;
} eos_conv_layer;

eos_conv_layer eos_alloc_conv_layer(size_t num_filters, size_t filter_shape);

void eos_conv_forward(eos_conv_layer *layer, eos_batch4f batch);
void eos_conv_backward(eos_conv_layer *layer, eos_batch4f gradients, float alpha);
size_t eos_conv_output_n_batch_size(eos_conv_layer *layer);
size_t eos_conv_output_n_rows(eos_conv_layer *layer);
size_t eos_conv_output_n_cols(eos_conv_layer *layer);
size_t eos_conv_output_n_channels(eos_conv_layer *layer);




#endif
