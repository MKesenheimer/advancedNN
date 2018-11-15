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
#define NDATASETS 4
#define NPARAMETERS (2*NINPUTS + NINPUTS * NNEURONS + NNEURONS + NNEURONS * NOUTPUTS + NOUTPUTS)

// choose parallelization
#define PARALLEL
#define NTHREADS 4

// choose transfer function
#define SIGMOID
//#define RELU
//#define TANH

// whether to use linesearch for optimization or not
#define LINESEARCH

// global variables
double lf; //global loss function to compare with at each step

struct ILayer {
    double weight;
    double theta;
};

struct Neuron {
    double weight[NINPUTS];
    double theta;
};

struct OLayer {
    double weight[NNEURONS];
    double theta;
};

struct NN {
    struct ILayer ilayer[NINPUTS];
    struct Neuron neuron[NNEURONS];
    struct OLayer olayer[NOUTPUTS];
    double output[NOUTPUTS];
};

struct DataSet {
    double xx[NINPUTS];
    double yy[NOUTPUTS];
};

double getRandomNumber();
void initNN(struct NN *nn);
void calculateNN(const double xx[], struct NN *nn);
double lossFunction(struct NN *nn, const struct DataSet dataset[]);
void train1(struct NN *nn, const struct DataSet dataset[], const double accuracy, const double learningrate);
void train2(struct NN *nn, const struct DataSet dataset[], const double accuracy, const double learningrate);

void structToArray(const struct NN *nn, double par[]);
void arrayToStruct(const double par[], struct NN *nn);


void snapshot(struct NN *nn);
void load(struct NN *nn);

#endif