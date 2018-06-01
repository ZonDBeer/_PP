prog := gemm3
prog_objs := gemm3.o

CC := gcc
LDFLAGS :=

CFLAGS := -Wall –fopenmp -o

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

gemm3.o: gemm3.c
clean:
	@rm -rf *.o $(prog)
