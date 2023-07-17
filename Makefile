CC = gcc
CFLAGS = -Wall -O0 -Wextra
INCLUDES = -I./include
TARGET = build/prog
CFILES = src/mnist.c src/tensor.c src/conv.c

all:
	$(CC) $(CFLAGS) $(CFILES) -o$(TARGET) $(INCLUDES) $(LIBS)