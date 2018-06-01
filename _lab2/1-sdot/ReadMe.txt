(sdot) Test CPU intel core i5 2500K 4.3GHz sse/avx:

// Default
Result (scalar): 134217728.000000 err = 465782272.000000
Elapsed time (scalar): 0.094508 sec.
Result (vectorized): 710521472.000000 err = 110521472.000000
Elapsed time (vectorized): 0.085457 sec.
Speedup: 1.11


//////////////////////////////////////////////////// With loop
For SSE version:
Result (scalar): 6000018.000000 err = 0.000000
Elapsed time (scalar): 0.001247 sec.
Result (vectorized): 6000018.000000 err = 0.000000
Elapsed time (vectorized): 0.001085 sec.
Speedup: 1.15


For AVX version:
Result (scalar): 6000018.000000 err = 0.000000
Elapsed time (scalar): 0.001094 sec.
Result (vectorized): 6000018.000000 err = 0.000000
Elapsed time (vectorized): 0.001346 sec.
Speedup: 0.81


/////////////////////////////////////////////////// Without loop

For SSE version:
Result (scalar): 6000018.000000 err = 0.000000
Elapsed time (scalar): 0.001195 sec.
Result (vectorized): 6000018.000000 err = 0.000000
Elapsed time (vectorized): 0.001171 sec.
Speedup: 1.02


For AVX version:
Result (scalar): 6000018.000000 err = 0.000000
Elapsed time (scalar): 0.001253 sec.
Result (vectorized): 6000018.000000 err = 0.000000
Elapsed time (vectorized): 0.001264 sec.
Speedup: 0.99

*****************************************************************

(sqrt) Test CPU intel core i5 6400 2.7GHz sse/avx
//////////////////////////////////////////////////// With loop

For SSE version:
Result (scalar): 6000018.000000 err = 0.000000
Elapsed time (scalar): 0.001556 sec.
Result (vectorized): 6000018.000000 err = 0.000000
Elapsed time (vectorized): 0.001514 sec.
Speedup: 1.03


For AVX version:
Result (scalar): 6000018.000000 err = 0.000000
Elapsed time (scalar): 0.001556 sec.
Result (vectorized): 6000018.000000 err = 0.000000
Elapsed time (vectorized): 0.001514 sec.
Speedup: 1.03


/////////////////////////////////////////////////// Without loop

For SSE version:
Result (scalar): 6000018.000000 err = 0.000000
Elapsed time (scalar): 0.001671 sec.
Result (vectorized): 6000018.000000 err = 0.000000
Elapsed time (vectorized): 0.000563 sec.
Speedup: 2.97


For AVX version:
Result (scalar): 6000018.000000 err = 0.000000
Elapsed time (scalar): 0.001522 sec.
Result (vectorized): 6000018.000000 err = 0.000000
Elapsed time (vectorized): 0.000500 sec.
Speedup: 3.04

