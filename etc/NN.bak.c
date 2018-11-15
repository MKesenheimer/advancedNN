#include "NN.h"

double rnd(double a, double b){
    double x = (double)rand()/(double)(RAND_MAX)*(b-a)+a;
    return x;
}

void structToArray(const struct NN *nn, double par[]) {
    int counter = 0;
    for(int i=0; i<NINPUTS; i++) {
        par[counter++] = nn->ilayer[i].theta;
        par[counter++] = nn->ilayer[i].weight;
    }

    for(int i=0; i<NNEURONS; i++) {
        par[counter++] = nn->neuron[i].theta;
        for(int n=0; n<NINPUTS; n++) {
            par[counter++] = nn->neuron[i].weight[n];
        }
    }
    
    for(int i=0; i<NOUTPUTS; i++) {
        par[counter++] = nn->olayer[i].theta;
        for(int n=0; n<NNEURONS; n++) {
            par[counter++] = nn->olayer[i].weight[n];
        }
    }
}

void arrayToStruct(const double par[], struct NN *nn) {
    int counter = 0;
    for(int i=0; i<NINPUTS; i++) {
        nn->ilayer[i].theta = par[counter++];
        nn->ilayer[i].weight = par[counter++];
    }

    for(int i=0; i<NNEURONS; i++) {
        nn->neuron[i].theta = par[counter++];
        for(int n=0; n<NINPUTS; n++) {
            nn->neuron[i].weight[n] = par[counter++];
        }
    }
    
    for(int i=0; i<NOUTPUTS; i++) {
        nn->olayer[i].theta = par[counter++];
        for(int n=0; n<NNEURONS; n++) {
            nn->olayer[i].weight[n] = par[counter++];
        }
    }
}

double transferFunction(double x, double theta) {
    double f;
    
    #ifdef SIGMOID
        f = 1/(1+exp(theta-x));
    #endif

    #ifdef RELU
        if(x+theta >= 0) {
            f = x+theta;
        } else {
            f = 0;
        }
    #endif

    #ifdef TANH
        f= tanh(theta+x);
    #endif

    return f;
}

void initNN(struct NN *nn) {
    // initialize input layer
    for(int i=0; i<NINPUTS; i++) {
        nn->ilayer[i].theta = 0;
        // the first layer has by default only one input per node
        nn->ilayer[i].weight = 0;
        nn->ilayer[i].input  = 0;
        for(int n=0; n<NNEURONS; n++) {
            nn->ilayer[i].output[n] = 0;
        }
    }

    // initialize neurons
    for(int i=0; i<NNEURONS; i++) {
        nn->neuron[i].theta = 0;
        for(int n=0; n<NINPUTS; n++) {
            nn->neuron[i].weight[n] = 0;
            nn->neuron[i].input[n]  = 0;
        }
        for(int n=0; n<NOUTPUTS; n++) {
            nn->neuron[i].output[n] = 0;
        }
    }
    
    // initialize output layer
    for(int i=0; i<NOUTPUTS; i++) {
        nn->olayer[i].theta = 0;
        for(int n=0; n<NNEURONS; n++) {
            nn->olayer[i].weight[n] = 0;
            nn->olayer[i].input[n]  = 0;
        }
        // the output layer has by default only one output per node
        nn->olayer[i].output = 0;
    }
}

void calculateNN(const double xx[], struct NN *nnn) {
    struct NN nn = *nnn;
    
    #ifdef PARALLEL1
        #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(xx) shared(nn)
    #endif
    for(int i=0; i<NINPUTS; i++) {
        double temp = nn.ilayer[i].weight*xx[i];
        for(int n=0; n<NNEURONS; n++) {
            nn.ilayer[i].output[n] = transferFunction(temp, nn.ilayer[i].theta);
        }
    }
    
    #ifdef PARALLEL1
        #pragma omp parallel for num_threads(NTHREADS) default(none) shared(nn)
    #endif
    for(int i=0; i<NNEURONS; i++) {
        double temp = 0;
        for(int n=0; n<NINPUTS; n++) {
            temp += nn.neuron[i].weight[n]*nn.ilayer[n].output[i];
        }
        for(int n=0; n<NOUTPUTS; n++) {
            nn.neuron[i].output[n] = transferFunction(temp, nn.neuron[i].theta);
        }
    }
    
    #ifdef PARALLEL1
        #pragma omp parallel for num_threads(NTHREADS) default(none) shared(nn)
    #endif
    for(int i=0; i<NOUTPUTS; i++) {
        double temp = 0;
        for(int n=0; n<NNEURONS; n++) {
            temp += nn.olayer[i].weight[n]*nn.neuron[n].output[i];
        }
        nn.olayer[i].output = transferFunction(temp, nn.olayer[i].theta);
    }

    *nnn = nn;
}

// calculate the output of the neural net with a small deviation h in the parameter ii
void calculateNNh(const double xx[], struct NN *nnn, const double h, const int ii) {
    struct NN nn = *nnn;
    double par[NPARAMETERS];
    structToArray(&nn,par);
    par[ii] = par[ii] + h;
    arrayToStruct(par,&nn);
    calculateNN(xx, &nn);
    *nnn = nn;
}

double costFunction(struct NN *nnn, const struct DataSet dataset[]) {
    double delta = 0;
    struct NN nn = *nnn;
    
    #ifdef PARALLEL2
        #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(dataset,nn) reduction(+:delta)
    #endif
    for(int i=0; i<NDATASETS; i++) {
        calculateNN(dataset[i].xx, &nn);
        double delta2 = 0;
        for(int j=0; j<NOUTPUTS; j++) {
            delta2 += pow(nn.olayer[j].output - dataset[i].yy[j],2);
        }
        delta += sqrt(delta2);
    }
    return delta/2;
}

double costFunction(const double par[], const struct DataSet dataset[]) {
    double delta = 0;
    
    #ifdef PARALLEL2
        #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(dataset,nn) reduction(+:delta)
    #endif
    for(int i=0; i<NDATASETS; i++) {
        calculateNN(dataset[i].xx, par);
        double delta2 = 0;
        for(int j=0; j<NOUTPUTS; j++) {
            delta2 += pow(nn.olayer[j].output - dataset[i].yy[j],2);
        }
        delta += sqrt(delta2);
    }
    return delta/2;
}

double costFunctionh(struct NN *nnn, const struct DataSet dataset[], const double h, const int ii) {
    double delta = 0;
    struct NN nn = *nnn;
    
    #ifdef PARALLEL2
        #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(dataset,nn) reduction(+:delta)
    #endif
    for(int i=0; i<NDATASETS; i++) {
        calculateNNh(dataset[i].xx, &nn, h, ii);
        double delta2 = 0;
        for(int j=0; j<NOUTPUTS; j++) {
            delta2 += pow(nn.olayer[j].output - dataset[i].yy[j],2);
        }
        delta += sqrt(delta2);
    }
    return delta/2;
}

void train(struct NN *nnn, const struct DataSet dataset[], const double accuracy, const double learningrate) {
    bool optimized = false;
    double h = 0.005;
    
    while (!optimized) {
        if(costFunction(nnn,dataset)<accuracy) {
            optimized = true;
        } else {
            // optimize the cost function
            struct NN nnhi, nnhj, nnhij;
            double derivThetaIn[NINPUTS];
            double derivWeightIn[NINPUTS];
            double derivThetaNe[NNEURONS];
            double derivWeightNe[NNEURONS][NINPUTS];
            double derivThetaOut[NOUTPUTS];
            double derivWeightOut[NOUTPUTS][NNEURONS];
            
            // second derivatives
            double hessThetaIn[NINPUTS][NINPUTS];
            double hessWeightIn[NINPUTS][NINPUTS];
            double hessThetaNe[NNEURONS][NNEURONS];
            double hessWeightNe[NNEURONS][NINPUTS][NNEURONS][NINPUTS];
            double hessThetaOut[NOUTPUTS][NOUTPUTS];
            double hessWeightOut[NOUTPUTS][NNEURONS][NOUTPUTS][NNEURONS];

            // calculate the derivatives
            struct NN nn = *nnn;
            nnhi = nn;
            nnhj = nn;
            nnhij = nn;
            
            #ifdef PARALLEL3
                #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(h,nn,dataset,nnhi,nnhj,nnhij) shared(derivThetaIn,derivWeightIn,hessThetaIn) //private(nnhi,nnhj,nnhij)
            #endif
            for(int i=0; i<NINPUTS; i++) {
                nnhi = nn;
                nnhi.ilayer[i].theta = nn.ilayer[i].theta + h;
                derivThetaIn[i] = (costFunction(&nnhi,dataset) - costFunction(&nn,dataset))/h;
                nnhi = nn;
                nnhi.ilayer[i].weight = nn.ilayer[i].weight + h;
                derivWeightIn[i] = (costFunction(&nnhi,dataset) - costFunction(&nn,dataset))/h;
                /*for (int j=i; j<NINPUTS; j++) {
                    nnhi = nn;
                    nnhi.ilayer[i].theta = nn.ilayer[i].theta + h;
                    nnhj = nn;
                    nnhj.ilayer[j].theta = nn.ilayer[j].theta + h;
                    nnhij = nn;
                    nnhij.ilayer[i].theta = nn.ilayer[i].theta + h;
                    nnhij.ilayer[j].theta = nn.ilayer[j].theta + h;
                    hessThetaIn[i][j] = (costFunction(&nnhij,dataset) - costFunction(&nnhi,dataset) - costFunction(&nnhj,dataset) + costFunction(&nn,dataset))/pow(h,2.0);
                    hessThetaIn[j][i] = hessThetaIn[i][j];
                }*/
            }
            #ifdef PARALLEL3
                #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(h,nn,dataset) shared(derivThetaNe,derivWeightNe) private(nnhi)
            #endif
            for(int i=0; i<NNEURONS; i++) {
                nnhi = nn;
                nnhi.neuron[i].theta = nn.neuron[i].theta + h;
                derivThetaNe[i] = (costFunction(&nnhi,dataset) - costFunction(&nn,dataset))/h;
                for(int n=0; n<NINPUTS; n++) {
                    nnhi = nn;
                    nnhi.neuron[i].weight[n] = nn.neuron[i].weight[n] + h;
                    derivWeightNe[i][n] = (costFunction(&nnhi,dataset) - costFunction(&nn,dataset))/h;
                }
            }
            #ifdef PARALLEL3
                #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(h,nn,dataset) shared(derivThetaOut,derivWeightOut) private(nnhi)
            #endif
            for(int i=0; i<NOUTPUTS; i++) {
                nnhi = nn;
                nnhi.olayer[i].theta = nn.olayer[i].theta + h;
                derivThetaOut[i] = (costFunction(&nnhi,dataset) - costFunction(&nn,dataset))/h;
                for(int n=0; n<NNEURONS; n++) {
                    nnhi = nn;
                    nnhi.olayer[i].weight[n] = nn.olayer[i].weight[n] + h;
                    derivWeightOut[i][n] = (costFunction(&nnhi,dataset) - costFunction(&nn,dataset))/h;
                }
            }
            
            // calculate the new parameters
            //#pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(learningrate) shared(derivThetaIn,derivWeightIn,nn)
            for(int i=0; i<NINPUTS; i++) {
                nn.ilayer[i].theta = nn.ilayer[i].theta - learningrate*derivThetaIn[i];
                nn.ilayer[i].weight = nn.ilayer[i].weight - learningrate*derivWeightIn[i];
            }
            //#pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(learningrate) shared(derivThetaNe,derivWeightNe,nn)
            for(int i=0; i<NNEURONS; i++) {
                nn.neuron[i].theta = nn.neuron[i].theta - learningrate*derivThetaNe[i];
                for(int n=0; n<NINPUTS; n++) {
                    nn.neuron[i].weight[n] = nn.neuron[i].weight[n] - learningrate*derivWeightNe[i][n];
                }
            }
            //#pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(learningrate) shared(derivThetaOut,derivWeightOut,nn)
            for(int i=0; i<NOUTPUTS; i++) {
                nn.olayer[i].theta = nn.olayer[i].theta - learningrate*derivThetaOut[i];
                for(int n=0; n<NNEURONS; n++) {
                    nn.olayer[i].weight[n] = nn.olayer[i].weight[n] - learningrate*derivWeightOut[i][n];
                }
            }
            
            *nnn = nn;
        }
    }
}

void trainh(struct NN *nnn, const struct DataSet dataset[], const double accuracy, const double learningrate) {
    bool optimized = false;
    double h = 0.005;
    
    while (!optimized) {
        if(costFunction(nnn,dataset)<accuracy) {
            optimized = true;
        } else {
            // optimize the cost function
            double deriv[NPARAMETERS];
            struct NN nn = *nnn;
            
            double par[NPARAMETERS];
            structToArray(&nn,par);
            
            // calculate the derivatives
            #ifdef PARALLEL3
                #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(h,nn,dataset) shared(deriv,par)
            #endif
            for(int i=0; i<NPARAMETERS; i++) {
                deriv[i] = (costFunctionh(&nn,dataset,h,i) - costFunction(&nn,dataset))/h;
                par[i] = par[i] - learningrate*deriv[i];
            }
            
            arrayToStruct(par,&nn);
            *nnn = nn;
        }
    }
}

void snapshot(struct NN *nn) {
    FILE *f1 = fopen("nn.dat", "w");
    if(f1 == NULL){
        printf("Error opening file!\n");
        exit(1);
    }

    for(int i = 0; i<NINPUTS; i++) {
        fprintf(f1,"w%i %f\n",i,nn->ilayer[i].weight);
    }
    for(int i = 0; i<NINPUTS; i++) {
        fprintf(f1,"t%i %f\n",i,nn->ilayer[i].theta);
    }
    for (int n = 0; n<NNEURONS; n++) {
        for(int i = 0; i<NINPUTS; i++) {
            fprintf(f1,"w%i%i %f\n",n,i,nn->neuron[n].weight[i]);
        }
    }
    for (int n = 0; n<NNEURONS; n++) {
        fprintf(f1,"t%i %f\n",n,nn->neuron[n].theta);
    }
    for (int n = 0; n<NOUTPUTS; n++) {
        for(int i = 0; i<NNEURONS; i++) {
            fprintf(f1,"w%i%i %f\n",n,i,nn->olayer[n].weight[i]);
        }
    }
    for (int n = 0; n<NOUTPUTS; n++) {
        fprintf(f1,"t%i %f\n",n,nn->olayer[n].theta);
    }
    fclose(f1);
}

void load(struct NN *nn) {
    FILE *f1 = fopen("nn.dat", "r");
    if(f1 == NULL){
        printf("File does not exist.\n");
        return;
    }

    char buff[255];
    char key1[255], key2[255];
    double val;

    for(int i = 0; i<NINPUTS; i++) {
        fscanf(f1, "%s", buff);
        strncpy(key1, buff, 255);
        fscanf(f1, "%s", buff);
        sscanf(buff, "%lf", &val);
        sprintf(key2, "%s%d", "w", i);
        if(strcmp(key1, key2)==0) {
            //printf("%s %f\n", key1, val);
            nn->ilayer[i].weight = val;
        }
    }
    for(int i = 0; i<NINPUTS; i++) {
        fscanf(f1, "%s", buff);
        strncpy(key1, buff, 255);
        fscanf(f1, "%s", buff);
        sscanf(buff, "%lf", &val);
        sprintf(key2, "%s%d", "t", i);
        if(strcmp(key1, key2)==0) {
            //printf("%s %f\n", key1, val);
            nn->ilayer[i].theta = val;
        }
    }
    for (int n = 0; n<NNEURONS; n++) {
        for(int i = 0; i<NINPUTS; i++) {
            fscanf(f1, "%s", buff);
            strncpy(key1, buff, 255);
            fscanf(f1, "%s", buff);
            sscanf(buff, "%lf", &val);
            sprintf(key2, "%s%d%d", "w", n, i);
            if(strcmp(key1, key2)==0) {
                //printf("%s %f\n", key1, val);
                nn->neuron[n].weight[i] = val;
            }
        }
    }
    for (int n = 0; n<NNEURONS; n++) {
        fscanf(f1, "%s", buff);
        strncpy(key1, buff, 255);
        fscanf(f1, "%s", buff);
        sscanf(buff, "%lf", &val);
        sprintf(key2, "%s%d", "t", n);
        if(strcmp(key1, key2)==0) {
            //printf("%s %f\n", key1, val);
            nn->neuron[n].theta = val;
        }
    }
    for (int n = 0; n<NOUTPUTS; n++) {
        for(int i = 0; i<NNEURONS; i++) {
            fscanf(f1, "%s", buff);
            strncpy(key1, buff, 255);
            fscanf(f1, "%s", buff);
            sscanf(buff, "%lf", &val);
            sprintf(key2, "%s%d%d", "w", n, i);
            if(strcmp(key1, key2)==0) {
                //printf("%s %f\n", key1, val);
                nn->olayer[n].weight[i] = val;
            }
        }
    }
    for (int n = 0; n<NOUTPUTS; n++) {
        fscanf(f1, "%s", buff);
        strncpy(key1, buff, 255);
        fscanf(f1, "%s", buff);
        sscanf(buff, "%lf", &val);
        sprintf(key2, "%s%d", "t", n);
        if(strcmp(key1, key2)==0) {
            //printf("%s %f\n", key1, val);
            nn->olayer[n].theta = val;
        }
    }

    fclose(f1);
}