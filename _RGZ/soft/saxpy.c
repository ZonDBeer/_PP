#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <sys/time.h>

#define EPS 1E-6

enum {
    n = 1000000
};

void saxpy(float *x, float *y, float a, int n)
{
    for (int i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
}

void saxpy_sse(float * restrict x, float * restrict y, float a, int n)
{
    __m128 *xx = (__m128 *)x;
    __m128 *yy = (__m128 *)y;
   
    int k = n / 4;
    __m128 aa = _mm_set1_ps(a);
    for (int i = 0; i < k; i++) {
        __m128 z = _mm_mul_ps(aa, xx[i]);          
        yy[i] = _mm_add_ps(z, yy[i]);
    }

    /* Loop reminder (n % 4 != 0) ? */
    for (int i = k * 4; i < n; i++)
		y[i] = a * x[i] + y[i]; 
}

void saxpy_avx(float * restrict x, float * restrict y, float a, int n)
{
    __m256 *xx = (__m256 *)x;
    __m256 *yy = (__m256 *)y;

    int k = n / 8;
    __m256 aa = _mm256_set1_ps(a);
    for (int i = 0; i < k; i++) {
        __m256 z = _mm256_mul_ps(aa, xx[i]);
        yy[i] = _mm256_add_ps(z, yy[i]);
    }

    /* Loop reminder (n % 8 != 0) ? */
    for (int i = k * 8; i < n; i++)
                y[i] = a * x[i] + y[i];
}

void daxpy_sse(double * restrict x, double * restrict y, double a, int n)
{
    __m128d *xx = (__m128d *)x;
    __m128d *yy = (__m128d *)y;
   
    int k = n / 2;
    __m128d aa = _mm_set1_pd(a);
    for (int i = 0; i < k; i++) {
        __m128d z = _mm_mul_pd(aa, xx[i]);          
        yy[i] = _mm_add_pd(z, yy[i]);
    }

    /* Loop reminder (n % 2 != 0) ? */
    for (int i = k * 2; i < n; i++)
		y[i] = a * x[i] + y[i]; 
}

void daxpy_avx(double * restrict x, double * restrict y, double a, int n)
{
    __m256d *xx = (__m256d *)x;
    __m256d *yy = (__m256d *)y;
   
    int k = n / 4;
    __m256d aa = _mm256_set1_pd(a);
    for (int i = 0; i < k; i++) {
        __m256d z = _mm256_mul_pd(aa, xx[i]);          
        yy[i] = _mm256_add_pd(z, yy[i]);
    }

    /* Loop reminder (n % 4 != 0) ? */
    for (int i = k * 4; i < n; i++)
		y[i] = a * x[i] + y[i]; 
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
    float *x, *y, a = 2.0;

    x = xmalloc(sizeof(*x) * n);
    y = xmalloc(sizeof(*y) * n);
    for (int i = 0; i < n; i++) {
        x[i] = i * 2 + 1.0;
        y[i] = i;
    }
    
    double t = wtime();
    saxpy(x, y, a, n);
    t = wtime() - t;    

    /* Verification */
    for (int i = 0; i < n; i++) {
        float xx = i * 2 + 1.0;
        float yy = a * xx + i;
        if (fabs(y[i] - yy) > EPS) {
            fprintf(stderr, "run_scalar: verification failed (y[%d] = %f != %f)\n", i, y[i], yy);
            break;
        }
    }
    
    printf("Elapsed time (scalar): %.6f sec.\n", t);
    free(x);
    free(y);    
    return t;
}


double run_vectorized()
{
    float *x, *y, a = 2.0;

    x = _mm_malloc(sizeof(*x) * n, 32);
    y = _mm_malloc(sizeof(*y) * n, 32);
    for (int i = 0; i < n; i++) {
        x[i] = i * 2 + 1.0;
        y[i] = i;
    }
    
    double t = wtime();
    //saxpy_sse(x, y, a, n);
    saxpy_avx(x, y, a, n);
    t = wtime() - t;
    
    /* Verification */
    for (int i = 0; i < n; i++) {
        float xx = i * 2 + 1.0;
        float yy = a * xx + i;
        if (fabs(y[i] - yy) > EPS) {
            fprintf(stderr, "run_vectorized: verification failed (y[%d] = %f != %f)\n", i, y[i], yy);
            break;
        }
    }
        
    printf("Elapsed time (vectorized): %.6f sec.\n", t);
    free(x);
    free(y);    
    return t;
}

double run_vectorized_double()
{
    double *x, *y, a = 2.0;

    x = _mm_malloc(sizeof(*x) * n, 64);
    y = _mm_malloc(sizeof(*y) * n, 64);
    for (int i = 0; i < n; i++) {
        x[i] = i * 2 + 1.0;
        y[i] = i;
    }
    
    double t = wtime();
    //daxpy_sse(x, y, a, n);
    daxpy_avx(x, y, a, n);
    t = wtime() - t;
    
    /* Verification */
    for (int i = 0; i < n; i++) {
        double xx = i * 2 + 1.0;
        double yy = a * xx + i;
        if (fabs(y[i] - yy) > EPS) {
            fprintf(stderr, "run_vectorized_double: verification failed (y[%d] = %f != %f)\n", i, y[i], yy);
            break;
        }
    }
        
    printf("Elapsed time (vectorized for double): %.6f sec.\n", t);
    free(x);
    free(y);    
    return t;
}

int main(int argc, char **argv)
{
    printf("SAXPY (y[i] = a * x[i] + y[i]; n = %d)\n", n);
    double tscalar = run_scalar();
    double tvec = run_vectorized();
    double tvec_d = run_vectorized_double();
    
    printf("Speedup: %.2f\n", tscalar / tvec);
    printf("Speedup for double: %.2f\n", tscalar / tvec_d);
        
    return 0;
}
