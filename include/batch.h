#include "tensor.h"

#ifndef _EOS_BATCH_H
#define _EOS_BATCH_H

typedef struct eos_batch4f
{
    eos_tensor3f *tensors;
    size_t count;
} eos_batch4f;

#endif