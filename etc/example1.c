//sudo port select --set gcc mp-gcc5
//gcc -I/opt/local/include/libomp -L/opt/local/lib/libomp/ -fopenmp example1.c -o example

#include <stdio.h>
#include <omp.h>
int main() {
  int i;
  #pragma omp parallel for
  for (i=1; i<1000; i++) {
    printf("%d\n",i);
  }
}
