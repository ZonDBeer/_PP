#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

void work(int size, int flag) {

    float **A = (float**)malloc(size * sizeof(float*));
    float **B = (float**)malloc(size * sizeof(float*));
    float **C = (float**)malloc(size * sizeof(float*));
    for ( int i = 0; i < size; i++ ) {
        A[i] = (float*)malloc(size * sizeof(float));
        B[i] = (float*)malloc(size * sizeof(float));
        C[i] = (float*)malloc(size * sizeof(float));
    }

    for (int i = 0; i < size; i++)
      for(int j = 0; j < size; j++) {
        A[i][j] = rand() % 10;
        B[i][j] = rand() % 10;
        C[i][j] = 0;
      }

    double t = wtime();
    #pragma omp parallel
    {
        //#pragma omp simd collapse(3)
        for (int i = 0; i < size; i++)
            for (int k = 0; k < size; k++)
            {
                #pragma omp simd
                for(int j = 0; j < size; j++ )
                C[i][j] += A[i][k] * B[k][j];
            }
    }
    t = wtime() - t;
    printf("%lf\n", t);

    for ( int i = 0; i < size; i++ ) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

}

int main( int argc, char **argv ) {
    //srand(time(NULL));
    printf("|A(row) B(row)|\n");
    for (int i = 256; i <= 4096; i+=256) {
        work(i, 1);
    }
    return 0;
}

