#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

#define EPS 1E-6

enum {
    n = 10000000
};

void init_particles(float *x, float *y, float *z, int n)
{
    for (int i = 0; i < n; i++) {
        x[i] = cos(i + 0.1);
        y[i] = cos(i + 0.2);
        z[i] = cos(i + 0.3);
    }
}

void distance(float *x, float *y, float *z, float *d, int n)
{   
    #pragma omp for simd 
    for (int i = 0; i < n; i++) {
        d[i] = sqrtf(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
    }
}

void *xmalloc(size_t size)
{
    void *p = malloc(size);
    if (!p) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }
    return p;
}

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

int main(int argc, char **argv)
{
    printf("Particles_OMP: n = %d)\n", n);
    
    float *d, *x, *y, *z;

    x = xmalloc(sizeof(*x) * n);
    y = xmalloc(sizeof(*y) * n);
    z = xmalloc(sizeof(*z) * n);
    d = xmalloc(sizeof(*d) * n);    
    
    init_particles(x, y, z, n);
    
    double t = 0;
    for (int r = 1000; r < 10000000; r = r * 10) {
      t = wtime();
      for (int iter = 0; iter < 100; iter++) {
          distance(x, y, z, d, r);
      }
      t = wtime() - t;  
      printf("%.6f sec.\n", t);
    }  

    
    free(x);
    free(y);    
    free(z);
    free(d);
        
    return 0;
}
