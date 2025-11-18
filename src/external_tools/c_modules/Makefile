# c_modules/Makefile
OS := $(shell uname)

ifeq ($(OS), Darwin)
    CC=gcc
    OUT=edr_macOS
else
    CC=gcc
    OUT=edr_linux
endif

all:
	$(CC) -o $(OUT) -fPIC -shared edr.c
