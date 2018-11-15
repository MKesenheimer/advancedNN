#include "NN.h"

double rnd(double a, double b){
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
    const double theta = 0.1;
    const double weight = 0.2;
    
    for(int i=0; i<NINPUTS; i++) {
        nn->ilayer[i].theta = rnd(0.0,theta);
        nn->ilayer[i].weight = rnd(0.0,weight);
    }

    for(int i=0; i<NNEURONS; i++) {
        nn->neuron[i].theta = rnd(0.0,theta);;
        for(int n=0; n<NINPUTS; n++) {
            nn->neuron[i].weight[n] = rnd(0.0,weight);
        }
    }
    
    for(int i=0; i<NOUTPUTS; i++) {
        nn->olayer[i].theta = rnd(0.0,theta);;
        for(int n=0; n<NNEURONS; n++) {
            nn->olayer[i].weight[n] = rnd(0.0,weight);
        }
        nn->output[i] = 0;
    }
}

void calculateNN(const double xx[], struct NN *nn) {
    double ioutput[NINPUTS][NNEURONS];
    double noutput[NNEURONS][NOUTPUTS];
    
    for(int i=0; i<NINPUTS; i++) {
        double temp = nn->ilayer[i].weight*xx[i];
        for(int n=0; n<NNEURONS; n++) {
            ioutput[i][n] = transferFunction(temp, nn->ilayer[i].theta);
        }
    }

    for(int n=0; n<NNEURONS; n++) {
        double temp = 0;
        for(int i=0; i<NINPUTS; i++) {
            temp += nn->neuron[n].weight[i]*ioutput[i][n];
        }
        for(int i=0; i<NOUTPUTS; i++) {
            noutput[n][i] = transferFunction(temp, nn->neuron[n].theta);
        }
    }
    
    for(int i=0; i<NOUTPUTS; i++) {
        double temp = 0;
        for(int n=0; n<NNEURONS; n++) {
            temp += nn->olayer[i].weight[n]*noutput[n][i];
        }
        nn->output[i] = transferFunction(temp, nn->olayer[i].theta);
    }
}

double lossFunction(struct NN *nn, const struct DataSet dataset[]) {
    double delta = 0;
    
    for(int i=0; i<NDATASETS; i++) {
        calculateNN(dataset[i].xx, nn);
        double delta2 = 0;
        for(int j=0; j<NOUTPUTS; j++) {
            delta2 += pow(nn->output[j] - dataset[i].yy[j],2);
        }
        delta += sqrt(delta2);
    }
    return delta/2;
}

double linesearch(const double par[], const double deriv[], const struct DataSet dataset[], const double learningrate) {
    struct NN nn2;
    double par2[NPARAMETERS];
    const double tau = 0.5, c = 0.5;
    double alpha = learningrate/tau;
    double m = norm(deriv,NPARAMETERS); //max(deriv,NPARAMETERS);
    double t = -c*m;
    double lf2;
    
    do {
        #ifdef PARALLEL
            #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(deriv,alpha,par) shared(par2)
        #endif
        for(int i=0; i<NPARAMETERS; i++) {
            par2[i] = par[i] - alpha*deriv[i];
        }
        arrayToStruct(par2,&nn2);
        lf2 = lossFunction(&nn2,dataset);
        alpha = tau*alpha;
        //printf("alpha*t = %f, lf-lf2 = %f\n",alpha*t,lf-lf2);
    } while(lf-lf2 < alpha*t);
    
    return alpha;
}

// train the network (gradient descent method)
void train1(struct NN *nn, const struct DataSet dataset[], const double accuracy, const double learningrate) {
    double h = 0.005;
    lf = lossFunction(nn,dataset);
    
    // optimize the cost function
    while(lf>accuracy) {
        struct NN nnh;
        double deriv[NPARAMETERS], temp;
            
        double par[NPARAMETERS];
        structToArray(nn,par);
        
        // calculate the derivatives
        #ifdef PARALLEL
            #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(h,dataset,par,lf) shared(deriv) private(temp,nnh)
        #endif
        for(int i=0; i<NPARAMETERS; i++) {
            temp = par[i];
            par[i] = par[i] + h;
            arrayToStruct(par,&nnh);
            deriv[i] = (lossFunction(&nnh,dataset) - lf)/h;
            par[i] = temp;
        }
        
        #ifdef LINESEARCH
            double alpha = linesearch(par,deriv,dataset,learningrate);
        #else
            double alpha = learningrate;
        #endif
        
        #ifdef PARALLEL
            #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(deriv,alpha) shared(par)
        #endif
        for(int i=0; i<NPARAMETERS; i++) {
            par[i] = par[i] - alpha*deriv[i];
        }
        arrayToStruct(par,nn);
        lf = lossFunction(nn,dataset);
        
        //printf("alpha = %f, lf = %f\n",alpha,lf);
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
            nn->olayer[n].theta = val;
        }
    }

    fclose(f1);
}