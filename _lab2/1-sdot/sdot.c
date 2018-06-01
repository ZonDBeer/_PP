#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <sys/time.h>

enum { n = 1000003 };

float sdot(float *x, float *y, int n)
{
    float s = 0;
    for (int i = 0; i < n; i++)
        s += x[i] * y[i];
    return s;
}

float sdot_sse(float *x, float *y, int n)
{
	__m128 *xx = (__m128 *) x;
	__m128 *yy = (__m128 *) y;
	
    float s = 0;
    float t[4] __attribute__ ((aligned (16)));
    int k = n / 4;
    for (int i = 0; i < k; i++) {
		__m128 tmp = _mm_mul_ps(xx[i], yy[i]);
		_mm_store_ps(t, tmp);
		for (int j = 0; j < 4; j++)
		  s += t[j];
		//s += t[0] + t[1] +t[2] +t[3];
	}
	
	for (int i = k * 4; i < n; i++)
        s += x[i] * y[i];
    return s;
}

float sdot_avx(float *x, float *y, int n)
{
	__m256 *xx = (__m256 *) x;
	__m256 *yy = (__m256 *) y;
	
    float s = 0;
    float t[8] __attribute__ ((aligned (32)));
    int k = n / 8;
    for (int i = 0; i < k; i++) {
		__m256 tmp = _mm256_mul_ps(xx[i], yy[i]);
		_mm256_store_ps(t, tmp);
		for (int j = 0; j < 8; j++)
		  s += t[j];
		//s += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
	}
	
	for (int i = k * 8; i < n; i++)
        s += x[i] * y[i];
    return s;
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

double run_scalar()
{
    float *x = xmalloc(sizeof(*x) * n);
    float *y = xmalloc(sizeof(*y) * n);
    for (int i = 0; i < n; i++) {
        x[i] = 2.0;
        y[i] = 3.0;
    }        
    
    double t = wtime();
    float res = sdot(x, y, n);
    t = wtime() - t;    
    
    float valid_result = 2.0 * 3.0 * n;
    printf("Result (scalar): %.6f err = %f\n", res, fabsf(valid_result - res));
    printf("Elapsed time (scalar): %.6f sec.\n", t);
    free(x);
    free(y);
    return t;
}

double run_vectorized()
{
    float *x = _mm_malloc(sizeof(*x) * n, 32);
    float *y = _mm_malloc(sizeof(*y) * n, 32);
    for (int i = 0; i < n; i++) {
        x[i] = 2.0;
        y[i] = 3.0;
    }        
    
    double t = wtime();
    //float res = sdot_sse(x, y, n);
    float res = sdot_avx(x, y, n);
    t = wtime() - t;    
    
    float valid_result = 2.0 * 3.0 * n;
    printf("Result (vectorized): %.6f err = %f\n", res, fabsf(valid_result - res));
    printf("Elapsed time (vectorized): %.6f sec.\n", t);
    free(x);
    free(y);
    return t;
}

int main(int argc, char **argv)
{
    printf("SDOT: n = %d\n", n);
    float tscalar = run_scalar();
    float tvec = run_vectorized();
    
    printf("Speedup: %.2f\n", tscalar / tvec);
        
    return 0;
}
