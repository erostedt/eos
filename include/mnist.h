#include "stdint.h"
#include "tensor.h"

#ifndef _MNIST_H
#define _MNIST_H

typedef struct mnist_t
{
    uint32_t num_datapoints;
    eos_tensor3f *features;
    uint8_t *targets; 
} mnist_t;


bool load_mnist(mnist_t *mnist, const char *features_path, const char *labels_path);

#endif