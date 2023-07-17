#include "stdint.h"
#include "tensor.h"

#ifndef _MNIST_H
#define _MNIST_H

typedef struct mnist_t
{
    uint32_t num_datapoints;
    Eos_Tensor3f *features;
    uint8_t *targets; 
} mnist_t;


bool load_mnist(mnist_t *mnist, const char *features_path, const char *labels_path);
void save_as_ppm(Eos_Tensor3f image, const char *file_path);

#endif