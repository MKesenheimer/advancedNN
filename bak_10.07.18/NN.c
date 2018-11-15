#include "NN.h"

double rnd(const double a, const double b){
    double x = (double)rand()/(double)(RAND_MAX)*(b-a)+a;
    return x;
}

double norm(const double p[], const double n) {
    double norm = 0;
    for(int i=0; i<n; i++) {
        norm += pow(p[i],2);
    }
    return pow(norm,0.5);
}

double max(const double p[], const double n) {
    double max = 0;
    for(int i=0; i<n; i++) {
        if(p[i]>max) max = p[i];
    }
    return max;
}

double transferFunction(const double x, const double theta) {
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

void initNN(union NN *nn) {
    const double theta = 0.1;
    const double weight = 0.2;
    
    for(int i=0; i<NINPUTS; i++) {
        nn->strct.ilayer[i].theta = rnd(0.0,theta);
        nn->strct.ilayer[i].weight = rnd(0.0,weight);
    }
    
    for(int i=0; i<NNEURONS; i++) {
        nn->strct.neuron[i].theta = rnd(0.0,theta);
        for(int n=0; n<NINPUTS; n++) {
            nn->strct.neuron[i].weight[n] = rnd(0.0,weight);
        }
    }
    
    for(int i=0; i<NOUTPUTS; i++) {
        nn->strct.olayer[i].theta = rnd(0.0,theta);
        for(int n=0; n<NNEURONS; n++) {
            nn->strct.olayer[i].weight[n] = rnd(0.0,weight);
        }
        nn->strct.output[i] = 0.0;
    }
}

void calculateNN(const double xx[], union NN *nn) {
    double ioutput[NINPUTS][NNEURONS];
    double noutput[NNEURONS][NOUTPUTS];
    
    for(int i=0; i<NINPUTS; i++) {
        double temp = nn->strct.ilayer[i].weight*xx[i];
        for(int n=0; n<NNEURONS; n++) {
            ioutput[i][n] = transferFunction(temp, nn->strct.ilayer[i].theta);
        }
    }
    
    for(int n=0; n<NNEURONS; n++) {
        double temp = 0;
        for(int i=0; i<NINPUTS; i++) {
            temp += nn->strct.neuron[n].weight[i]*ioutput[i][n];
        }
        for(int i=0; i<NOUTPUTS; i++) {
            noutput[n][i] = transferFunction(temp, nn->strct.neuron[n].theta);
        }
    }
    
    for(int i=0; i<NOUTPUTS; i++) {
        double temp = 0;
        for(int n=0; n<NNEURONS; n++) {
            temp += nn->strct.olayer[i].weight[n]*noutput[n][i];
        }
        nn->strct.output[i] = transferFunction(temp, nn->strct.olayer[i].theta);
    }
}

double lossFunction(union NN *nn, const struct DataSet dataset[]) {
    double delta = 0;
    for(int i=0; i<NDATASETS; i++) {
        calculateNN(dataset[i].xx, nn);
        double delta2 = 0;
        for(int j=0; j<NOUTPUTS; j++) {
            delta2 += pow(nn->strct.output[j] - dataset[i].yy[j],2);
        }
        delta += sqrt(delta2);
    }
    return delta/2;
}

double linesearch(union NN *nn, const double deriv[], const struct DataSet dataset[], const double learningrate) {
    union NN nn2;
    const double tau = 0.5, c = 0.5;
    double alpha = learningrate/tau;
    double m = norm(deriv,NPARAMETERS);
    double t = -c*m;
    double lf2;
    
    do {
        #ifdef PARALLEL
            #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(deriv,alpha) shared(nn,nn2)
        #endif
        for(int i=0; i<NPARAMETERS; i++) {
            nn2.par[i] = nn->par[i] - alpha*deriv[i];
        }
        lf2 = lossFunction(&nn2,dataset);
        alpha = tau*alpha;
    } while(nn->strct.lf - lf2 < alpha*t);
    
    return alpha;
}

// train the network (gradient descent method)
void train1(union NN *nn, const struct DataSet dataset[], const double accuracy, const double learningrate) {
    const double h = 0.005;
    union NN nnh = *nn;
    
    nn->strct.lf = lossFunction(nn,dataset);
    
    // optimize the cost function
    while(nn->strct.lf > accuracy) {
        double deriv[NPARAMETERS], temp;
        
        // calculate the derivatives
        #ifdef PARALLEL
            #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(h,dataset,nn,nnh) shared(deriv) private(temp)
        #endif
        for(int i=0; i<NPARAMETERS; i++) {
            nnh = *nn;
            nnh.par[i] = nn->par[i] + h;
            deriv[i] = (lossFunction(&nnh,dataset) - nn->strct.lf)/h;
        }
        
        #ifdef LINESEARCH
            double alpha = linesearch(nn,deriv,dataset,learningrate);
        #else
            double alpha = learningrate;
        #endif
        
        #ifdef PARALLEL
            #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(deriv,alpha) shared(nn)
        #endif
        for(int i=0; i<NPARAMETERS; i++) {
            nn->par[i] = nn->par[i] - alpha*deriv[i];
        }
        nn->strct.lf = lossFunction(nn,dataset);
        
        //printf("alpha = %f, lf = %f\n",alpha,nn->strct.lf);
    }
}

void snapshot(union NN *nn) {
    FILE *f1 = fopen("nn.dat", "w");
    if(f1 == NULL){
        printf("Error opening file!\n");
        exit(1);
    }

    for(int i = 0; i<NINPUTS; i++) {
        fprintf(f1,"w%i %f\n",i,nn->strct.ilayer[i].weight);
    }
    for(int i = 0; i<NINPUTS; i++) {
        fprintf(f1,"t%i %f\n",i,nn->strct.ilayer[i].theta);
    }
    for (int n = 0; n<NNEURONS; n++) {
        for(int i = 0; i<NINPUTS; i++) {
            fprintf(f1,"w%i%i %f\n",n,i,nn->strct.neuron[n].weight[i]);
        }
    }
    for (int n = 0; n<NNEURONS; n++) {
        fprintf(f1,"t%i %f\n",n,nn->strct.neuron[n].theta);
    }
    for (int n = 0; n<NOUTPUTS; n++) {
        for(int i = 0; i<NNEURONS; i++) {
            fprintf(f1,"w%i%i %f\n",n,i,nn->strct.olayer[n].weight[i]);
        }
    }
    for (int n = 0; n<NOUTPUTS; n++) {
        fprintf(f1,"t%i %f\n",n,nn->strct.olayer[n].theta);
    }
    fclose(f1);
}

void load(union NN *nn) {
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
            nn->strct.ilayer[i].weight = val;
        }
    }
    for(int i = 0; i<NINPUTS; i++) {
        fscanf(f1, "%s", buff);
        strncpy(key1, buff, 255);
        fscanf(f1, "%s", buff);
        sscanf(buff, "%lf", &val);
        sprintf(key2, "%s%d", "t", i);
        if(strcmp(key1, key2)==0) {
            nn->strct.ilayer[i].theta = val;
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
                nn->strct.neuron[n].weight[i] = val;
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
            nn->strct.neuron[n].theta = val;
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
                nn->strct.olayer[n].weight[i] = val;
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
            nn->strct.olayer[n].theta = val;
        }
    }

    fclose(f1);
}