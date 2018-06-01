prog := dgemv
prog_objs := dgemv.o

CC := gcc
CFLAGS := -std=c99 -g -Wall -O2 -fopenmp
LDFLAGS := -fopenmp

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

dgemv.o: dgemv.c

clean:
	@rm -rf *.o $(prog)
