#include "tensor.h"
#include "batch.h"

#ifndef _EOS_CONV_H
#define _EOS_CONV_H
typedef struct Eos_Conv_Layer
{
    Eos_Batch4f batch;
    Eos_Batch4f filters;
    
    size_t filter_rows;
    size_t filter_cols;
    size_t filter_depth;

    Eos_Batch4f filter_gradients;
    Eos_Batch4f local_gradients;
} Eos_Conv_Layer;

Eos_Conv_Layer eos_alloc_conv_layer(size_t num_filters, size_t filter_shape);

void eos_conv_forward(Eos_Conv_Layer *layer, Eos_Batch4f batch);
void eos_conv_backward(Eos_Conv_Layer *layer, Eos_Batch4f gradients, float alpha);
size_t eos_conv_output_n_batch_size(Eos_Conv_Layer *layer);
size_t eos_conv_output_n_rows(Eos_Conv_Layer *layer);
size_t eos_conv_output_n_cols(Eos_Conv_Layer *layer);
size_t eos_conv_output_n_channels(Eos_Conv_Layer *layer);




#endif
