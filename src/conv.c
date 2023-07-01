#include "math.h"
#include "conv.h"
#include "assert.h"
#include "stdint.h"


void eos_sigmoid_activate(eos_batch4f inputs)
{
    for (int i = 0; i < inputs.count; i++)
    {
        eos_tensor3f input_tensor = inputs.tensors[i];
        for (int j = 0; j < input_tensor.rows * input_tensor.cols * input_tensor.channels; j++)
        {
            input_tensor.data[j] = 1./(1+expf(-input_tensor.data[j]));
        }
    }
}

void eos_relu_activate(eos_batch4f inputs)
{
    for (int i = 0; i < inputs.count; i++)
    {
        eos_tensor3f input_tensor = inputs.tensors[i];
        for (int j = 0; j < input_tensor.rows * input_tensor.cols * input_tensor.channels; j++)
        {
            if (input_tensor.data[j] < 0)
                input_tensor.data[j] = 0.0f;
        }
    }
}


void eos_sigmoid_dactivate(eos_batch4f inputs, eos_batch4f dinputs)
{
    for (int i = 0; i < inputs.count; i++)
    {
        eos_tensor3f input_tensor = inputs.tensors[i];
        eos_tensor3f dinput_tensor = dinputs.tensors[i];
        for (int j = 0; j < input_tensor.rows * input_tensor.cols * input_tensor.channels; j++)
        {
            float s = 1./(1+expf(-input_tensor.data[j]));
            dinput_tensor.data[j] *= s * (1.0f - s);
        }
    }
}


void eos_relu_dactivate(eos_batch4f inputs, eos_batch4f dinputs)
{
    for (int i = 0; i < inputs.count; i++)
    {
        eos_tensor3f input_tensor = inputs.tensors[i];
        eos_tensor3f dinput_tensor = dinputs.tensors[i];
        for (int j = 0; j < input_tensor.rows * input_tensor.cols * input_tensor.channels; j++)
        {
            if (input_tensor.data[j] < 0)
                dinput_tensor.data[j] = 0.0f;
        }
    }
}


void eos_activate(Activation activation, eos_batch4f inputs)
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

void eos_dactivate(Activation activation, eos_batch4f inputs, eos_batch4f dinputs)
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


void eos_conv_forward(eos_conv_layer *layer, eos_batch4f inputs, eos_batch4f outputs)
{
    // ADD BOUNDS CHECK
    for (int i = 0; i < inputs.count; i++)
    {
        eos_tensor3f input_tensor = inputs.tensors[i];
        eos_tensor3f output_tensor = outputs.tensors[i];
        for (int j = 0; j < layer->filters.count; j++)
        {
            eos_tensor3f filter = layer->filters.tensors[j];
            for (int k = 0; k < output_tensor.rows; k++)
            {
                for (int l = 0; l < output_tensor.cols; l++)
                {
                    for (int m = 0; m < filter.rows; m++)
                    {
                        for (int n = 0; n < filter.cols; n++)
                        {
                            for (int c = 0; c < input_tensor.channels; c++)
                            {
                                for (int s = 0; s < layer->stride_rows; s++)
                                {
                                    for (int t = 0; t < layer->stride_cols; t++)
                                    {
                                        output_tensor.data[((j * output_tensor.rows + k) * output_tensor.cols + l) * input_tensor.channels + c] += input_tensor.data[((k * layer->stride_rows + m) * input_tensor.cols + l * layer->stride_cols + n) * input_tensor.channels + c] * filter.data[(m * filter.cols + n) * input_tensor.channels + c];
                                    }
                                }
                            }
                        }
                    }
                    output_tensor.data[((j * output_tensor.rows + k) * output_tensor.cols + l) * input_tensor.channels] += layer->biases[j];
                }
            }
        }
    }
    eos_activate(layer->activation, outputs);
}


void eos_conv_backward(eos_conv_layer *layer, eos_batch4f inputs, eos_batch4f incoming_gradients, eos_batch4f filter_gradients, float *dbiases, float alpha, eos_batch4f outgoing_gradients)
{
    // ADD BOUNDS CHECK
    // Inputs should be activated inputs.
    // RETURN BY VALUE BY ALLOCATION IN FUNCTION USING ARENA?
    for (int i = 0; i < inputs.count; i++)
    {
        eos_tensor3f input_tensor = inputs.tensors[i];
        eos_tensor3f outgoing_tensor = outgoing_gradients.tensors[i];
        eos_tensor3f_zero(outgoing_tensor);
        eos_tensor3f incoming_tensor = incoming_gradients.tensors[i];

        for (int j = 0; j < layer->filters.count; j++)
        {
            eos_tensor3f filter_tensor = layer->filters.tensors[j];
            eos_tensor3f dfilter_tensor = filter_gradients.tensors[j];
            eos_tensor3f_zero(dfilter_tensor);
            dbiases[j] = 0.0f;

            for (int k = 0; k < incoming_tensor.rows; k++)
            {
                for (int l = 0; l < incoming_tensor.cols; l++)
                {
                    for (int m = 0; m < filter_tensor.rows; m++)
                    {
                        for (int n = 0; n < filter_tensor.cols; n++)
                        {
                            for (int c = 0; c < input_tensor.channels; c++)
                            {
                                for (int s = 0; s < layer->stride_rows; s++)
                                {
                                    for (int t = 0; t < layer->stride_cols; t++)
                                    {
                                        float scale = incoming_tensor.data[((j * incoming_tensor.rows + k) * incoming_tensor.cols + l) * input_tensor.channels + c];
                                        dfilter_tensor.data[(m * filter_tensor.cols + n) * input_tensor.channels + c] += scale * input_tensor.data[((k * layer->stride_rows + m) * input_tensor.cols + l * layer->stride_cols + n) * input_tensor.channels + c];   
                                        outgoing_tensor.data[((k * layer->stride_rows + m) * outgoing_tensor.cols + l * layer->stride_cols + n) * outgoing_tensor.channels + c] += scale * filter_tensor.data[(m * filter_tensor.cols + n) * input_tensor.channels + c];
                                    }
                                }
                            }
                        }
                    }
                    dbiases[j] += incoming_tensor.data[((j * incoming_tensor.rows + k) * incoming_tensor.cols + l) * input_tensor.channels];
                }
            }
        }
    }

    eos_dactivate(layer->activation, inputs, outgoing_gradients);

    for (int i = 0; i < layer->filters.count; i++)
    {
        eos_tensor3f filter = layer->filters.tensors[i];
        eos_tensor3f dfilter = filter_gradients.tensors[i];
        for (int j = 0; j < filter.rows * filter.cols * filter.channels; j++)
        {
            filter.data[j] -= alpha * dfilter.data[j];
        }
        layer->biases[i] -= alpha * dbiases[i];
    }
}


int eos_conv_output_n_rows(eos_conv_layer *layer, int input_rows)
{
    return (input_rows - layer->filter_rows) / layer->stride_rows + 1;
}

int eos_conv_output_n_cols(eos_conv_layer *layer, int input_cols)
{    
    int output_width = (input_cols - layer->filter_cols) / layer->stride_cols + 1;
}

int eos_conv_output_n_channels(eos_conv_layer *layer)
{
    return layer->filters.count;
}
