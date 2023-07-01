#include "tensor.h"
#include "batch.h"

#ifndef _EOS_CONV_H
#define _EOS_CONV_H

typedef enum 
{
    NONE,
    SIGMOID,
    RELU,
} Activation;

typedef struct eos_conv_layer
{
    eos_batch4f filters;
    float *biases;
    
    size_t filter_rows;
    size_t filter_cols;
    size_t filter_depth;
    size_t stride_rows;
    size_t stride_cols;
    Activation activation;
} eos_conv_layer;

eos_conv_layer eos_alloc_conv_layer(size_t num_filters, size_t filter_shape);

void eos_conv_forward(eos_conv_layer *layer, eos_batch4f inputs, eos_batch4f outputs);
void eos_conv_backward(eos_conv_layer *layer, eos_batch4f inputs, eos_batch4f incoming_gradients, eos_batch4f filter_gradients, float *dbiases, float alpha, eos_batch4f outgoing_gradients);
int eos_conv_output_n_rows(eos_conv_layer *layer, int input_rows);
int eos_conv_output_n_cols(eos_conv_layer *layer, int input_cols);
int eos_conv_output_n_channels(eos_conv_layer *layer);




#endif
