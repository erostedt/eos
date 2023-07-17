#include "stdlib.h"
#include "stdbool.h"
#include "mnist.h"
#include "conv.h"
#include "tensor.h"


int32_t main()
{   
    const char *features = "./mnist/t10k-images.idx3-ubyte";
    const char *targets = "./mnist/t10k-labels.idx1-ubyte";
    
    mnist_t mnist;
    bool success = load_mnist(&mnist, features, targets);
    if (!success)
        return 1;

    size_t idx = 0;
    Eos_Tensor3f image = mnist.features[idx];
    save_as_ppm(image, "image.ppm");

    //E eos_alloc_conv_layer(3, 3);


    return 0;
}