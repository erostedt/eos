#include "activation.h"
#include "math.h"
#include "assert.h"
#include "stdint.h"
#include "stdlib.h"


void eos_sigmoid_forward(Eos_Batch4f inputs)
{
    for (size_t i = 0; i < inputs.count; i++)
    {
        Eos_Tensor3f input_tensor = inputs.tensors[i];
        for (size_t j = 0; j < input_tensor.rows * input_tensor.cols * input_tensor.channels; j++)
        {
            input_tensor.data[j] = 1./(1+expf(-input_tensor.data[j]));
        }
    }
}

void eos_relu_forward(Eos_Batch4f inputs)
{
    for (size_t i = 0; i < inputs.count; i++)
    {
        Eos_Tensor3f input_tensor = inputs.tensors[i];
        for (size_t j = 0; j < input_tensor.rows * input_tensor.cols * input_tensor.channels; j++)
        {
            if (input_tensor.data[j] < 0)
                input_tensor.data[j] = 0.0f;
        }
    }
}


void eos_sigmoid_backward(Eos_Batch4f inputs, Eos_Batch4f dinputs)
{
    for (size_t i = 0; i < inputs.count; i++)
    {
        Eos_Tensor3f input_tensor = inputs.tensors[i];
        Eos_Tensor3f dinput_tensor = dinputs.tensors[i];
        for (size_t j = 0; j < input_tensor.rows * input_tensor.cols * input_tensor.channels; j++)
        {
            float s = 1./(1+expf(-input_tensor.data[j]));
            dinput_tensor.data[j] *= s * (1.0f - s);
        }
    }
}


void eos_relu_backward(Eos_Batch4f inputs, Eos_Batch4f dinputs)
{
    for (size_t i = 0; i < inputs.count; i++)
    {
        Eos_Tensor3f input_tensor = inputs.tensors[i];
        Eos_Tensor3f dinput_tensor = dinputs.tensors[i];
        for (size_t j = 0; j < input_tensor.rows * input_tensor.cols * input_tensor.channels; j++)
        {
            if (input_tensor.data[j] < 0)
                dinput_tensor.data[j] = 0.0f;
        }
    }
}


void eos_act_forward(Activation activation, Eos_Batch4f inputs)
{
    switch (activation)
    {
        case NONE:
            break;

        case SIGMOID:
            eos_sigmoid_activate(inputs);
            break;
        
        case RELU:
            eos_relu_activate(inputs);
            break;

        default:
            assert(0);
            break;
    }
}

void eos_act_backward(Activation activation, Eos_Batch4f inputs, Eos_Batch4f dinputs)
{
    switch (activation)
    {
        case NONE:
            break;
            
        case SIGMOID:
            eos_sigmoid_dactivate(inputs, dinputs);
            break;
        
        case RELU:
            eos_relu_dactivate(inputs, dinputs);
            break;

        default:
            assert(0);
            break;
    }
}