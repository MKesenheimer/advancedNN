#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#include "NN.h"

#define NTEAMS NINPUTS

#define RUSSLAND 0
#define SAUDI_ARABIEN 1
#define AGYPTEN 2
#define URUGUAY 3

#define PORTUGAL 4
#define SPANIEN 5
#define MAROKKO 6
#define IRAN 7

#define FRANKREICH 8
#define AUSTRALIEN 9
#define PERU 10
#define DANEMARK 11

#define ARGENTINIEN 12
#define ISLAND 13
#define KROATION 14
#define NIGERIA 15

#define BRASILIEN 16
#define SCHWEIZ 17
#define COSTA_RICA 18
#define SERBIEN 19

#define DEUTSCHLAND 20
#define MEXIKO 21
#define SCHWEDEN 22
#define SUDKOREA 23

#define BELGIEN 24
#define PANAMA 25
#define TUNESIEN 26
#define ENGLAND 27

#define POLEN 28
#define SENEGAL 29
#define KOLUMBIEN 30
#define JAPAN 31

int main(int argc, char *argv[]) {
    
    // random numbers
    srand((unsigned int)time(NULL));
    
    // initialize NN
    union NN nn;
    initNN(&nn);
    
    // data sets
    struct DataSet dataset[NDATASETS];
    
    // initialize datasets
    for (int j=0; j<NDATASETS; j++) {
        for (int i=0; i<NOUTPUTS; i++) {
            dataset[j].yy[i] = 0;
        }
        for (int i=0; i<NINPUTS; i++) {
            dataset[j].xx[i] = 0;
        }
    }

    // dataset 1
    #if NDATASETS >= 1
        dataset[0].xx[RUSSLAND] = 1;
        dataset[0].xx[SAUDI_ARABIEN] = 1;
        dataset[0].yy[0] = 0.5;
        dataset[0].yy[1] = 0.0;
    #endif
    
    // dataset 2
    #if NDATASETS >= 2
        dataset[1].xx[AGYPTEN] = 1;
        dataset[1].xx[URUGUAY] = 1;
        dataset[1].yy[0] = 0.0;
        dataset[1].yy[1] = 0.1;
    #endif
    
    // dataset 3
    #if NDATASETS >= 3
        dataset[2].xx[MAROKKO] = 1;
        dataset[2].xx[IRAN] = 1;
        dataset[2].yy[0] = 0.0;
        dataset[2].yy[1] = 0.1;
    #endif
    
    // dataset 4
    #if NDATASETS >= 4
        dataset[3].xx[PORTUGAL] = 1;
        dataset[3].xx[SPANIEN] = 1;
        dataset[3].yy[0] = 0.3;
        dataset[3].yy[1] = 0.3;
    #endif
    
    // dataset 5
    #if NDATASETS >= 5
        dataset[4].xx[FRANKREICH] = 1;
        dataset[4].xx[AUSTRALIEN] = 1;
        dataset[4].yy[0] = 0.2;
        dataset[4].yy[1] = 0.1;
    #endif
    
    // dataset 6
    #if NDATASETS >= 6
        dataset[5].xx[ARGENTINIEN] = 1;
        dataset[5].xx[ISLAND] = 1;
        dataset[5].yy[0] = 0.1;
        dataset[5].yy[1] = 0.1;
    #endif
    
    // load previous parameters
    //load(&nn);
    // train the network
    //train1(&nn, dataset, 0.1, 1);
    train2(&nn, dataset, 0.1, 1);

    // initialize test sample
    double xx[NTEAMS];
    for (int i=0; i<NTEAMS; i++) {
        xx[i] = 0;
    }
    
    // calculate output
    xx[RUSSLAND] = 1;
    xx[SAUDI_ARABIEN] = 1;
    calculateNN(xx, &nn);
    printf("Russland      = %.0f\n",nn.strct.output[0]*10);
    printf("Saudi Arabien = %.0f\n\n",nn.strct.output[1]*10);
    
    // save snapshot
    snapshot(&nn);

	return 0;
}
