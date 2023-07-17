#include "math.h"
#include "conv.h"
#include "assert.h"
#include "stdint.h"
#include "stdlib.h"


float eos_sigmoid(float x)
{
    return 1./(1+expf(-x));
}


float eos_deriv_sigmoid(float x)
{
    float s = eos_sigmoid(x);
    return s * (1.0f - s);
}


float eos_relu(float x)
{
    return (x > 0.0f) ? x : 0.0f;
}


float eos_deriv_relu(float x)
{
    return (x > 0.0f) ? 1.0f : 0.0f;
}


void eos_conv_batched_cross_corr(eos_batch4f dst, eos_batch4f batch, eos_batch4f filters)
{ 
    assert((dst.count == filters.count));
    for (size_t tensor_idx = 0; tensor_idx < batch.count; tensor_idx++)
    {
        eos_tensor3f tensor = batch.tensors[tensor_idx];
        eos_tensor3f feature_maps = dst.tensors[tensor_idx]; 
        for (size_t filter_idx = 0; filter_idx < filters.count; filter_idx++)
        {
            eos_tensor3f filter = filters.tensors[filter_idx];
            assert((filter.rows <= tensor.rows) && (filter.cols <= tensor.cols) && (filter.channels == tensor.channels));
            float sum;
            for (size_t imrow = 0; imrow < tensor.rows-filter.rows + 1; imrow++)
            {
                for (size_t imcol = 0; imcol < tensor.cols-filter.cols + 1; imcol++)
                {
                    sum = 0;
                    for (size_t channel = 0; channel < tensor.channels; channel++)
                    {
                        for (size_t frow = 0; frow < filter.rows; frow++)
                        {
                            for (size_t fcol = 0; fcol < filter.cols; fcol++)
                                sum += TENSOR_AT(tensor, imrow+frow, imcol+fcol, channel) * TENSOR_AT(filter, frow, fcol, channel);
                        }
                    }
                    TENSOR_AT(feature_maps, imrow, imcol, filter_idx) = sum;
                }
            }
        }
    }

}


void eos_conv_clear_gradients(eos_conv_layer *layer)
{
    for (size_t filter_idx = 0; layer->filter_gradients.count; filter_idx++)
    {
        eos_tensor3f filter_gradients = layer->filter_gradients.tensors[filter_idx];
        for (size_t row = 0; row < filter_gradients.rows; row++)
        {
            for (size_t col = 0; col < filter_gradients.cols; col++)
            {
                for (size_t channel = 0; channel < filter_gradients.channels; channel++)
                {
                    TENSOR_AT(filter_gradients, row, col, channel) = 0.0f;
                }
            }
        }
    }

    for (size_t gradient_idx = 0; gradient_idx < layer->local_gradients.count; gradient_idx++)
    {
        eos_tensor3f gradients = layer->local_gradients.tensors[gradient_idx];
        for (size_t row = 0; row < gradients.rows; row++)
        {
            for (size_t col = 0; col < gradients.cols; col++)
            {
                for (size_t channel = 0; channel < gradients.channels; channel++)
                {
                    TENSOR_AT(gradients, row, col, channel) = 0.0f;
                }
            }                
        }
    }
}

void eos_conv_fill_filter_gradients(eos_conv_layer *layer, eos_batch4f gradients)
{
    eos_conv_batched_cross_corr(layer->filter_gradients, layer->batch, gradients);
}

void eos_conv_apply_filter_gradients(eos_conv_layer *layer, float alpha)
{
    for (size_t filter_idx = 0; filter_idx < layer->filter_gradients.count; filter_idx++)
    {
        eos_tensor3f filter = layer->filters.tensors[filter_idx];
        eos_tensor3f gradients = layer->filter_gradients.tensors[filter_idx];
        assert((filter.rows == gradients.rows) && (filter.cols == gradients.cols) && (filter.channels == gradients.channels));
        for (size_t row = 0; row < filter.rows; row++)
        {
            for (size_t col = 0; col < filter.cols; col++)
            {
                for (size_t channel = 0; channel < filter.channels; channel++)
                {
                    TENSOR_AT(filter, row, col, channel) -= alpha * TENSOR_AT(gradients, row, col, channel);
                }
            }
        }
    }
}

void eos_conv_fill_local_gradients(eos_conv_layer *layer, eos_batch4f gradients)
{
    // See if this can be sped up.
    for (size_t tensor_idx = 0; tensor_idx < gradients.count; tensor_idx++)
    {
        eos_tensor3f inputs = layer->batch.tensors[tensor_idx];
        eos_tensor3f incoming_gradients = gradients.tensors[tensor_idx];
        eos_tensor3f local_gradients = layer->local_gradients.tensors[tensor_idx]; 
        int inrows = (int)inputs.rows;
        int incols = (int)inputs.cols;
        int outrows = (int)incoming_gradients.rows;
        int outcols = (int)incoming_gradients.cols;
        int filter_rows = (int)layer->filter_rows;
        int filter_cols = (int)layer->filter_cols;
        
        int row_diff, col_diff;
        for (int inrow = 0; inrow < inrows; inrow++)
        {
            for (int incol = 0; incol < incols; incol++)
            {
                for (size_t filter_idx = 0; filter_idx < layer->filters.count; filter_idx++)
                {
                    eos_tensor3f filter = layer->filters.tensors[filter_idx];
                    float sum = 0;
                    for (int outrow = 0; outrow < outrows; outrow++)
                    {
                        row_diff = inrow - outrow;
                        if ((row_diff < 0) || row_diff >= filter_rows)
                            continue;
                    
                        for (int outcol = 0; outcol < outcols; outcol++)
                        {
                            col_diff = incol - outcol;    
                            if ((col_diff >= 0) && (col_diff < filter_cols))
                            {
                                sum += TENSOR_AT(filter, row_diff, col_diff, filter_idx) * TENSOR_AT(incoming_gradients, outrow, outcol, filter_idx);
                            }
                        }
                    }
                    TENSOR_AT(local_gradients, inrow, incol, filter_idx) = sum;
                }
            }
        }
    }
}


void eos_conv_forward(eos_conv_layer *layer,  eos_batch4f batch)
{
    assert(batch.count <= layer->filter_gradients.count);
    assert((layer->filter_rows <= batch.tensors->rows) && (layer->filter_cols <= batch.tensors->cols) && (layer->filter_depth == batch.tensors->channels));
    
    layer->batch.count = batch.count;
    eos_conv_batched_cross_corr(layer->batch, batch, layer->filters);
}



void eos_conv_backward(eos_conv_layer *layer, eos_batch4f gradients, float alpha)
{
    eos_conv_clear_gradients(layer);
    eos_conv_fill_local_gradients(layer, gradients);
    eos_conv_fill_filter_gradients(layer, gradients);
    eos_conv_apply_filter_gradients(layer, alpha);
}   



size_t eos_conv_output_n_batch_size(eos_conv_layer *layer)
{
    return layer->batch.count;
}

size_t eos_conv_output_n_rows(eos_conv_layer *layer)
{
    assert (layer->batch.count > 0);
    return layer->batch.tensors[0].rows - layer->filter_rows + 1;
}

size_t eos_conv_output_n_cols(eos_conv_layer *layer)
{
    assert (layer->batch.count > 0);
    return layer->batch.tensors[0].cols - layer->filter_cols + 1;
}

size_t eos_conv_output_n_channels(eos_conv_layer *layer)
{
    return layer->filters.count;
}
