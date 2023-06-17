CC = gcc
CFLAGS = -Wall -O0 -Wextra
INCLUDES = -I./include -I./third_party/raylib/include
TARGET = build/prog
CFILES = src/mnist.c src/tensor.c src/conv.c

LIBS=-L./third_party/raylib/libs -lraylib

all:
	$(CC) $(CFLAGS) $(CFILES) -o$(TARGET) $(INCLUDES) $(LIBS)