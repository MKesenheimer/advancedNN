//sudo port select --set gcc mp-gcc5
//gcc -I/opt/local/include/libomp -L/opt/local/lib/libomp/ -fopenmp example4.c -o example

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
    for (int j=0; j<100000; j++) {

        const int n = 1000;
        int   i;
        float a[n], b[n], sum;

        for (i=0; i < n; i++)
            a[i] = b[i] = i * 1.0;
        sum = 0.0;

        #pragma omp parallel for reduction(+:sum)
        for (i=0; i < n; i++)
            sum = sum + (a[i] * b[i]);

        printf("   Sum = %f\n",sum);
    }

}