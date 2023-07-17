#include "tensor.h"

#ifndef _EOS_BATCH_H
#define _EOS_BATCH_H

typedef struct Eos_Batch4f
{
    Eos_Tensor3f *tensors;
    size_t count;
} Eos_Batch4f;

Eos_Batch4f eos_batch_alloc_contigious(size_t count, size_t height, size_t width, size_t channels);
Eos_Batch4f eos_batch_alloc_spread(size_t count, size_t height, size_t width, size_t channels);
void eos_batch_free_spread(Eos_Batch4f *batch);
void eos_batch_free_contigious(Eos_Batch4f *batch);
#endif