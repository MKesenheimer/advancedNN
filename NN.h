#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>


#define NINPUTS 32
#define NOUTPUTS 2
#define NNEURONS 5
#define NDATASETS 3
#define NPARAMETERS (2*NINPUTS + NINPUTS * NNEURONS + NNEURONS + NNEURONS * NOUTPUTS + NOUTPUTS)

// choose parallelization
//#define PARALLEL
//#define NTHREADS 4

// choose transfer function
#define SIGMOID
//#define RELU
//#define TANH

// whether to use linesearch for optimization or not
//#define LINESEARCH

struct NNstruct {
    double iweight[NINPUTS];
    double itheta[NINPUTS];
    double nweight[NNEURONS][NINPUTS];
    double ntheta[NNEURONS];
    double oweight[NOUTPUTS][NNEURONS];
    double otheta[NOUTPUTS];
    double output[NOUTPUTS];
    double lf;
};

union NN {
    struct NNstruct strct;
    double par[NPARAMETERS + NOUTPUTS + 1];
};

struct DataSet {
    double xx[NINPUTS];
    double yy[NOUTPUTS];
};

void initNN(union NN *nn);
void calculateNN(const double xx[], union NN *nn);
void train1(union NN *nn, const struct DataSet dataset[], const double accuracy, const double learningrate);
void train2(union NN *nn, const struct DataSet dataset[], const double accuracy, const double learningrate);

void snapshot(union NN *nn);
void load(union NN *nn);

#endif