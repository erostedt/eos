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

typedef struct Eos_Conv_Layer
{
    Eos_Batch4f filters;
    float *biases;
    
    size_t filter_rows;
    size_t filter_cols;
    size_t filter_depth;
    size_t stride_rows;
    size_t stride_cols;
    Activation activation;
} Eos_Conv_Layer;

Eos_Conv_Layer eos_alloc_conv_layer(size_t num_filters, size_t filter_shape);

void eos_conv_forward(Eos_Conv_Layer *layer, Eos_Batch4f inputs, Eos_Batch4f outputs);
void eos_conv_backward(Eos_Conv_Layer *layer, Eos_Batch4f inputs, Eos_Batch4f incoming_gradients, Eos_Batch4f filter_gradients, float *dbiases, float alpha, Eos_Batch4f outgoing_gradients);
int eos_conv_output_n_rows(Eos_Conv_Layer *layer, int input_rows);
int eos_conv_output_n_cols(Eos_Conv_Layer *layer, int input_cols);
int eos_conv_output_n_channels(Eos_Conv_Layer *layer);

#endif