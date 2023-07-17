#include "batch.h"

Eos_Batch4f eos_batch_alloc_contigious(size_t count, size_t height, size_t width, size_t channels)
{
    float *data = (float*)malloc(count * height * width * channels * sizeof(float));

    Eos_Tensor3f *features = (Eos_Tensor3f*)malloc(count * sizeof(Eos_Tensor3f));
    size_t stepsize = height * width * channels;
    for (size_t index = 0; index < count; index++)
        features[index] = eos_tensor3f_borrow(height, width, channels, data + (index * stepsize));
    return (Eos_Batch4f) {.tensors=features, .count=count};
}

Eos_Batch4f eos_batch_alloc_spread(size_t count, size_t height, size_t width, size_t channels)
{
    Eos_Tensor3f *features = (Eos_Tensor3f*)malloc(count * sizeof(Eos_Tensor3f));
    size_t stepsize = height * width * channels;
    for (size_t index = 0; index < count; index++)
        features[index] = eos_tensor3f_alloc(height, width, channels);
    return (Eos_Batch4f) {.tensors=features, .count=count};
}


void eos_batch_free_spread(Eos_Batch4f *batch)
{
    for (size_t i = 0; i < batch->count; i++)
    {
        eos_tensor3f_free(&batch->tensors[i]);
    }
    batch->count = 0;
    batch = NULL;
}

void eos_batch_free_contigious(Eos_Batch4f *batch)
{
    if (batch->count > 0)
    {
        eos_tensor3f_free(&batch->tensors[0]);
    }

    for (size_t i = 0; i < batch->count; i++)
    {
        free(&batch->tensors[i]);
    }
    batch->count = 0;
    batch = NULL;

}