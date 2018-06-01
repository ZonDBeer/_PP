prog := gemm1
prog_objs := gemm1.o

CC := gcc
LDFLAGS :=

CFLAGS := -Wall -O3

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

gemm1.o: gemm1.c
clean:
	@rm -rf *.o $(prog)
