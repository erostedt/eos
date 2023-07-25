#include "tensor.h"
#include "batch.h"

#ifndef _EOS_ACT_H
#define _EOS_ACT_H

typedef enum 
{
    NONE,
    SIGMOID,
    RELU,
} Activation;

void eos_sigmoid_forward(Eos_Batch4f inputs);
void eos_relu_forward(Eos_Batch4f inputs);
void eos_sigmoid_backward(Eos_Batch4f inputs, Eos_Batch4f dinputs);
void eos_relu_backward(Eos_Batch4f inputs, Eos_Batch4f dinputs);
void eos_act_forward(Activation activation, Eos_Batch4f inputs);
void eos_act_backward(Activation activation, Eos_Batch4f inputs, Eos_Batch4f dinputs);
#endif