#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <sys/time.h>

#define EPS 1E-6

enum {
    n = 100000007
};

void compute_sqrt(float *in, float *out, int n)
{
    for (int i = 0; i < n; i++) {
        if (in[i] > 0)
            out[i] = sqrtf(in[i]);
        else
            out[i] = 0.0;
    }
}

void compute_sqrt_sse(float *in, float *out, int n)
{
	__m128 *xx = (__m128 *) in;
	__m128 *yy = (__m128 *) out;
	__m128 zero = _mm_setzero_ps();
	
	int k = n / 4;	
	for (int i = 0; i < k; i++) {
		
		__m128 tmp = _mm_cmp_ps(xx[i], zero, 14);
		yy[i] = _mm_sqrt_ps(tmp);        
    }
    
    for (int i = k * 4; i < n; i++) {
        if (in[i] > 0)
            out[i] = sqrtf(in[i]);
        else
            out[i] = 0.0;
    }
}

void compute_sqrt_avx(float *in, float *out, int n)
{
	__m256 *xx = (__m256 *) in;
	__m256 *yy = (__m256 *) out;
	__m256 zero = _mm256_setzero_ps();
	
	int k = n / 8;
	for (int i = 0; i < k; i++) {		
		__m256 tmp = _mm256_cmp_ps(xx[i], zero, 14);
		yy[i] = _mm256_sqrt_ps(tmp);        
    }
    
    for (int i = k * 8; i < n; i++) {
        if (in[i] > 0)
            out[i] = sqrtf(in[i]);
        else
            out[i] = 0.0;
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

double run_scalar()
{
    float *in = xmalloc(sizeof(*in) * n);
    float *out = xmalloc(sizeof(*out) * n);
    srand(0);
    for (int i = 0; i < n; i++) {
        in[i] = rand() > RAND_MAX / 2 ? 0 : rand() / (float)RAND_MAX * 1000.0;
    }
    double t = wtime();
    compute_sqrt(in, out, n);
    t = wtime() - t;    
    
#if 0
    for (int i = 0; i < n; i++)
        printf("%.4f ", out[i]);
    printf("\n");        
#endif
    
    printf("Elapsed time (scalar): %.6f sec.\n", t);
    free(in);
    free(out);
    return t;
}

double run_vectorized()
{
    float *in = _mm_malloc(sizeof(*in) * n, 32);
    float *out = _mm_malloc(sizeof(*out) * n, 32);
    srand(0);
    for (int i = 0; i < n; i++) {
        in[i] = rand() > RAND_MAX / 2 ? 0 : rand() / (float)RAND_MAX * 1000.0;
    }
    double t = wtime();
	//compute_sqrt_sse(in, out, n);
    compute_sqrt_avx(in, out, n);
    t = wtime() - t;    

#if 0
    for (int i = 0; i < n; i++)
        printf("%.4f ", out[i]);
    printf("\n");        
#endif
    
    for (int i = 0; i < n; i++) {
        float r = in[i] > 0 ? sqrtf(in[i]) : 0.0;
        if (fabs(out[i] - r) > EPS) {
            fprintf(stderr, "Verification: FAILED at out[%d] = %f != %f\n", i, out[i], r);
            break;
        }
    }
    
    printf("Elapsed time (vectorized): %.6f sec.\n", t);
    free(in);
    free(out);
    return t;
}

int main(int argc, char **argv)
{
    printf("Tabulate sqrt: n = %d\n", n);
    double tscalar = run_scalar();
    double tvec = run_vectorized();
    
    printf("Speedup: %.2f\n", tscalar / tvec);
        
    return 0;
}
