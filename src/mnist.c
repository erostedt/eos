#include "stdlib.h"
#include "stdio.h"
#include "stdbool.h"
#include "mnist.h"
#include "assert.h"


typedef struct Mnist_Features_Header
{
    int32_t datatype;
    int32_t dims;
    int32_t num_datapoints;
    int32_t num_rows;
    int32_t num_cols;
} Mnist_Features_Header;

typedef struct Mnist_Labels_Header
{
    int32_t datatype;
    int32_t dims;
    int32_t num_datapoints;
} Mnist_Labels_Header;
 

void _flip_int(int32_t *integer)
{
    uint8_t *byte = (uint8_t*)integer;
    uint8_t temp;
    temp = *(byte);
    *(byte) = *(byte + 3);
    *(byte + 3) = temp;

    byte++;

    temp = *(byte);
    *(byte) = *(byte + 1);
    *(byte + 1) = temp;
}

Mnist_Features_Header _read_feature_header(FILE *fp)
{
    Mnist_Features_Header header;
    fgetc(fp);
    fgetc(fp);
    header.datatype = fgetc(fp);
    header.dims = fgetc(fp);
    fread(&header.num_datapoints, sizeof(int32_t), 1, fp);
    fread(&header.num_rows, sizeof(int32_t), 1, fp);
    fread(&header.num_cols, sizeof(int32_t), 1, fp);
    _flip_int(&header.num_datapoints);
    _flip_int(&header.num_rows);
    _flip_int(&header.num_cols);
    return header;
}

Mnist_Labels_Header _read_label_header(FILE *fp)
{
    Mnist_Labels_Header header;
    fgetc(fp);
    fgetc(fp);
    header.datatype = fgetc(fp);
    header.dims = fgetc(fp);
    
    int32_t num_datapoints;
    fread(&header.num_datapoints, sizeof(num_datapoints), 1, fp);
    _flip_int(&header.num_datapoints);
    return header;
}

float *_read_mnist_pixels(FILE *fp, Mnist_Features_Header header)
{
    float *pixels = (float*)malloc(header.num_datapoints * header.num_rows * header.num_cols * sizeof(float));
    size_t elems_per_feature = header.num_rows * header.num_cols;
    size_t curr_pixel = 0;
    float scale = 1.f/255.f;
    for (size_t index = 0; index < (size_t)header.num_datapoints; index++)
    {
        for (size_t pixel_index = 0; pixel_index < elems_per_feature; pixel_index++)
            pixels[curr_pixel++] = scale * ((float)fgetc(fp));
        
    }
    return pixels;
}

uint8_t *_read_mnist_targets(FILE *fp, Mnist_Labels_Header header)
{
    uint8_t *targets = (uint8_t*)malloc(header.num_datapoints * sizeof(uint8_t));
    for (size_t index = 0; index < (size_t)header.num_datapoints; index++)
        targets[index] = (uint8_t)fgetc(fp);
    
    return targets;

}

bool load_mnist(mnist_t *mnist, const char *features_path, const char *labels_path)
{
    FILE *feature_fp = fopen(features_path, "rb");
    if (feature_fp == NULL)
    {
        printf("Could not load features_path.");
        return false;
    }

    FILE *labels_fp = fopen(labels_path, "rb");
    if (labels_fp == NULL)
    {
        fclose(feature_fp);
        printf("Could not load labels path.");
        return false;
    }

    Mnist_Features_Header feature_header = _read_feature_header(feature_fp);
    Mnist_Labels_Header label_header = _read_label_header(labels_fp);

    if (feature_header.datatype != label_header.datatype)
    {
        printf("Datatype of features and targets are different.");
        fclose(feature_fp);
        fclose(labels_fp);
        return false;
    }

    if (feature_header.num_datapoints != label_header.num_datapoints)
    {
        printf("Number of features and targets are different.");
        fclose(feature_fp);
        fclose(labels_fp);
        return false;
    }

    float *normalized_pixels = _read_mnist_pixels(feature_fp, feature_header);
    fclose(feature_fp);
    
    uint8_t *targets = _read_mnist_targets(labels_fp, label_header);
    fclose(labels_fp);

    Eos_Tensor3f *features = (Eos_Tensor3f*)malloc(feature_header.num_datapoints * sizeof(Eos_Tensor3f));
    size_t stepsize = feature_header.num_cols * feature_header.num_rows;
    for (size_t index = 0; index < (size_t)feature_header.num_datapoints; index++)
        features[index] = eos_tensor3f_borrow(feature_header.num_rows, feature_header.num_cols, 1, normalized_pixels + (index * stepsize));
    
    mnist->num_datapoints = feature_header.num_datapoints;
    mnist->features=features;
    mnist->targets=targets;
    return true;
}

/*
Texture2D eos_tensor3f_grayscale_texture_alloc(eos_tensor3f tensor)
{
    return LoadTextureFromImage((Image)
    {
        .data = tensor.data,
        .width = tensor.cols,
        .height = tensor.rows,
        .format = PIXELFORMAT_UNCOMPRESSED_R32,
        .mipmaps = 1
    });
}
*/

void save_as_ppm(Eos_Tensor3f image, const char *file_path)
{
    assert(image.channels == 1);
    FILE *f = fopen(file_path, "wb");
    if (f == NULL) 
    {
        fprintf(stderr, "ERROR: could not open file %s: %m\n", file_path);
        exit(1);
    }

    fprintf(f, "P6\n%zu %zu 255\n", image.cols, image.rows);

    for (size_t y = 0; y < image.rows; ++y) 
    {
        for (size_t x = 0; x < image.cols; ++x) 
        {
            float fpixel = TENSOR_AT(image, y, x, 0);
            uint8_t pixel = fpixel * 255.0;
            uint8_t buf[3] = {pixel, pixel, pixel};
            
            fwrite(buf, sizeof(buf), 1, f);
        }
    }

    fclose(f);
}