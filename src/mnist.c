#include "stdlib.h"
#include "stdio.h"
#include "stdint.h"
#include "stdbool.h"
#include "raylib.h" 
#include "mnist.h"


typedef struct mnist_features_header
{
    int32_t datatype;
    int32_t dims;
    int32_t num_datapoints;
    int32_t num_rows;
    int32_t num_cols;
} _mnist_features_header;

typedef struct mnist_labels_header
{
    int32_t datatype;
    int32_t dims;
    int32_t num_datapoints;
} _mnist_labels_header;
 

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

_mnist_features_header _read_feature_header(FILE *fp)
{
    _mnist_features_header header;
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

_mnist_labels_header _read_label_header(FILE *fp)
{
    _mnist_labels_header header;
    fgetc(fp);
    fgetc(fp);
    header.datatype = fgetc(fp);
    header.dims = fgetc(fp);
    
    int32_t num_datapoints;
    fread(&header.num_datapoints, sizeof(num_datapoints), 1, fp);
    _flip_int(&header.num_datapoints);
    return header;
}

float *_read_mnist_pixels(FILE *fp, _mnist_features_header header)
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

uint8_t *_read_mnist_targets(FILE *fp, _mnist_labels_header header)
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

    _mnist_features_header feature_header = _read_feature_header(feature_fp);
    _mnist_labels_header label_header = _read_label_header(labels_fp);

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

    eos_tensor3f *features = (eos_tensor3f*)malloc(feature_header.num_datapoints * sizeof(eos_tensor3f));
    size_t stepsize = feature_header.num_cols * feature_header.num_rows;
    for (size_t index = 0; index < (size_t)feature_header.num_datapoints; index++)
        features[index] = eos_tensor3f_borrow(feature_header.num_rows, feature_header.num_cols, 1, normalized_pixels + (index * stepsize));
    
    mnist->num_datapoints = feature_header.num_datapoints;
    mnist->features=features;
    mnist->targets=targets;
    return true;
}

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

int32_t main()
{   
    const int32_t WINDOW_WIDTH = 100;
    const int32_t WINDOW_HEIGHT = 100;
    const char *WINDOW_NAME = "MNIST IMAGE";
    
    const char *features = ".\\mnist\\t10k-images.idx3-ubyte";
    const char *targets = ".\\mnist\\t10k-labels.idx1-ubyte";
    
    mnist_t mnist;
    bool success = load_mnist(&mnist, features, targets);
    if (!success)
        return 1;

    SetTraceLogLevel(LOG_ERROR);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME);

    SetTargetFPS(60);

    size_t idx = 3;
    eos_tensor3f image = mnist.features[idx];

    Texture2D pre = eos_tensor3f_grayscale_texture_alloc(image);
    printf("%d", mnist.targets[idx]);
    while (!WindowShouldClose())
    {
        BeginDrawing();
        DrawTexture(pre, 0, 0, WHITE);
        EndDrawing();
    };

    UnloadTexture(pre);
    CloseWindow();
    return 0;
}