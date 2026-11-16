CC ?= gcc
CFLAGS = -O3 -march=native -ffast-math -std=c99 -Wall -Wextra -fPIC
LDFLAGS = -shared -lm

TARGET = libgoshawk.so
SRC = goshawk_core.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
