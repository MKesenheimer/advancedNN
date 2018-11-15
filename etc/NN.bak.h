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
//#define PARALLEL1
//#define PARALLEL2
#define PARALLEL3
#define NTHREADS 4

// choose transfer function
#define SIGMOID
//#define RELU
//#define TANH

struct ILayer {
    double input;
    double weight;
    double output[NNEURONS];
    double theta;
};

struct Neuron {
    double input[NINPUTS];
    double weight[NINPUTS];
    double output[NOUTPUTS];
    double theta;
};

struct OLayer {
    double input[NNEURONS];
    double weight[NNEURONS];
    double output;
    double theta;
};

struct NN {
    struct ILayer ilayer[NINPUTS];
    struct Neuron neuron[NNEURONS];
    struct OLayer olayer[NOUTPUTS];
};

struct DataSet {
    double xx[NINPUTS];
    double yy[NOUTPUTS];
};

double getRandomNumber();
void initNN(struct NN *nn);
void calculateNN(const double xx[], struct NN *nn);
void calculateNNh(const double xx[], struct NN *nn, const double h, const int ii);
double costFunction(struct NN *nn, const struct DataSet dataset[]);
double costFunctionh(struct NN *nn, const struct DataSet dataset[], const double h, const int ii);
void train(struct NN *nn, const struct DataSet dataset[], const double accuracy, const double learningrate);
void trainh(struct NN *nn, const struct DataSet dataset[], const double accuracy, const double learningrate);

void structToArray(const struct NN *nn, double par[]);
void arrayToStruct(const double par[], struct NN *nn);


void snapshot(struct NN *nn);
void load(struct NN *nn);

#endif