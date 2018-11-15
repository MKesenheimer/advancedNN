//sudo port select --set gcc mp-gcc5
//gcc -I/opt/local/include/libomp -L/opt/local/lib/libomp/ -fopenmp example3.c -o example

/******************************************************************************
* FILE: omp_hello.c
* DESCRIPTION:
*   OpenMP Example - Hello World - C/C++ Version
*   In this simple example, the master thread forks a parallel region.
*   All threads in the team obtain their unique thread number and print it.
*   The master thread only prints the total number of threads.  Two OpenMP
*   library routines are used to obtain the number of threads and each
*   thread's number.
* AUTHOR: Blaise Barney  5/99
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main (int argc, char *argv[]) {
    int nthreads, tid;
    const int size = 256;
    double sinTable[size];
    
    #pragma omp parallel for
    for(int n=0; n<size; ++n) {
        /* Obtain thread number */
        tid = omp_get_thread_num();
        printf("Hello World from thread = %d\n", tid);
        sinTable[n] = sin(2 * M_PI * n / size);
    }
}
