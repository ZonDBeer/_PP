prog := gemm2
prog_objs := gemm2.o

CC := gcc
LDFLAGS :=

CFLAGS := -Wall –fopenmp -o

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

gemm2.o: gemm2.c
clean:
	@rm -rf *.o $(prog)
