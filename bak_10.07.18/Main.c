#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#include "NN.h"

const int NTEAMS = NINPUTS;

const int RUSSLAND = 0;
const int SAUDI_ARABIEN = 1;
const int AGYPTEN = 2;
const int URUGUAY = 3;

const int PORTUGAL = 4;
const int SPANIEN = 5;
const int MAROKKO = 6;
const int IRAN = 7;

const int FRANKREICH = 8;
const int AUSTRALIEN = 9;
const int PERU = 10;
const int DANEMARK = 11;

const int ARGENTINIEN = 12;
const int ISLAND = 13;
const int KROATION = 14;
const int NIGERIA = 15;

const int BRASILIEN = 16;
const int SCHWEIZ = 17;
const int COSTA_RICA = 18;
const int SERBIEN = 19;

const int DEUTSCHLAND = 20;
const int MEXIKO = 21;
const int SCHWEDEN = 22;
const int SUDKOREA = 23;

const int BELGIEN = 24;
const int PANAMA = 25;
const int TUNESIEN = 26;
const int ENGLAND = 27;

const int POLEN = 28;
const int SENEGAL = 29;
const int KOLUMBIEN = 30;
const int JAPAN = 31;

int main(int argc, char *argv[]) {
    
    // random numbers
    srand(time(NULL));
    
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
    train1(&nn, dataset, 0.05, 6);

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
