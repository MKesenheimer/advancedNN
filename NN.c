#include "NN.h"
#pragma warning(disable:4996)

double rnd(const double a, const double b) {
    double x = (double)rand()/(double)(RAND_MAX)*(b-a)+a;
    return x;
}

double dot(const double p1[], const double p2[]) {
    double dot = 0;
    for(int i=0; i<NPARAMETERS; i++) {
        dot += p1[i]*p2[i];
    }
    return dot;
}

double norm(const double p[]) {
    double norm = 0;
    for(int i=0; i<NPARAMETERS; i++) {
        norm += pow(p[i],2);
    }
    return pow(norm,0.5);
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
        nn->strct.itheta[i] = rnd(0.0,theta);
        nn->strct.iweight[i] = rnd(0.0,weight);
    }
    
    for(int n=0; n<NNEURONS; n++) {
        nn->strct.ntheta[n] = rnd(0.0,theta);
        for(int i=0; i<NINPUTS; i++) {
            nn->strct.nweight[n][i] = rnd(0.0,weight);
        }
    }
    
    for(int i=0; i<NOUTPUTS; i++) {
        nn->strct.otheta[i] = rnd(0.0,theta);
        for(int n=0; n<NNEURONS; n++) {
            nn->strct.oweight[i][n] = rnd(0.0,weight);
        }
        nn->strct.output[i] = 0.0;
    }
}

void calculateNN(const double xx[], union NN *nn) {
    double ioutput[NINPUTS][NNEURONS];
    double noutput[NNEURONS][NOUTPUTS];
    
    for(int i=0; i<NINPUTS; i++) {
        double temp = nn->strct.iweight[i]*xx[i];
        for(int n=0; n<NNEURONS; n++) {
            ioutput[i][n] = transferFunction(temp, nn->strct.itheta[i]);
        }
    }
    
    for(int n=0; n<NNEURONS; n++) {
        double temp = 0;
        for(int i=0; i<NINPUTS; i++) {
            temp += nn->strct.nweight[n][i]*ioutput[i][n];
        }
        for(int i=0; i<NOUTPUTS; i++) {
            noutput[n][i] = transferFunction(temp, nn->strct.ntheta[n]);
        }
    }
    
    for(int i=0; i<NOUTPUTS; i++) {
        double temp = 0;
        for(int n=0; n<NNEURONS; n++) {
            temp += nn->strct.oweight[i][n]*noutput[n][i];
        }
        nn->strct.output[i] = transferFunction(temp, nn->strct.otheta[i]);
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

double linesearch(union NN *nn, const double grad[], const double p[], const struct DataSet dataset[], const double learningrate) {
    union NN nn2;
    const double tau = 0.5, c = 0.5;
    double alpha = learningrate/tau;
    double m = dot(grad,p);//norm(p);
    double t = -c*m;
    double lf = nn->strct.lf, lf2;
    const int max_iter = 10;
    for(int i=0; i<max_iter; i++) {
        #ifdef PARALLEL
            #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(dir,alpha) shared(nn,nn2)
        #endif
        for(int i=0; i<NPARAMETERS; i++) {
            nn2.par[i] = nn->par[i] + alpha*p[i];
        }
        lf2 = lossFunction(&nn2,dataset);
        alpha = tau*alpha;
        //printf("lf = %f, lf2 = %f, alpha*t = %f\n", lf,lf2,alpha*t);
        if(lf - lf2 >= alpha*t) break; // Wolfe condition
    }
    //exit(0);
    return alpha;
}

// train the network (gradient descent method)
void train1(union NN *nn, const struct DataSet dataset[], const double accuracy, const double learningrate) {
    const double h = 0.005;
    union NN nnh = *nn;
    
    nn->strct.lf = lossFunction(nn,dataset);
    
    // optimize the cost function
    while(nn->strct.lf > accuracy) {
        double grad[NPARAMETERS], p[NPARAMETERS];
        
        // calculate the gradatives
        #ifdef PARALLEL
            #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(h,dataset,nn,nnh) shared(grad,p) private(temp)
        #endif
        for(int i=0; i<NPARAMETERS; i++) {
            nnh = *nn;
            nnh.par[i] = nn->par[i] + h;
            grad[i] = (lossFunction(&nnh,dataset) - nn->strct.lf)/h;
            p[i] = -grad[i];
        }
        
        // normalize the search direction
        double nrm = norm(p);
        for (int i=0; i<NPARAMETERS; i++) {
            p[i] = p[i]/nrm;
        }
        
        #ifdef LINESEARCH
            double alpha = linesearch(nn,grad,p,dataset,learningrate);
        #else
            double alpha = learningrate;
        #endif
        
        #ifdef PARALLEL
            #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(p,alpha) shared(nn)
        #endif
        for(int i=0; i<NPARAMETERS; i++) {
            nn->par[i] = nn->par[i] + alpha*p[i];
        }
        nn->strct.lf = lossFunction(nn,dataset);
        
        printf("alpha = %f, lf = %f, gradf = %f\n",alpha,nn->strct.lf,norm(grad));
    }
}

// train the network (quasi newtonian method)
void train2(union NN *nn, const struct DataSet dataset[], const double accuracy, const double learningrate) {
    const double h = 0.001;
    union NN nnh = *nn;
    double p[NPARAMETERS], s[NPARAMETERS], y[NPARAMETERS];
    double grad[NPARAMETERS], grad2[NPARAMETERS];
    double hess[NPARAMETERS][NPARAMETERS];
    
    for (int i=0; i<NPARAMETERS; i++) {
        grad[i] = 0.0;
        grad2[i] = 0.0;
        p[i] = 0.0;
        for (int j=0; j<NPARAMETERS; j++) {
            hess[i][j] = 0.0;
        }
        hess[i][i] = 1.0;
    }
    
    nn->strct.lf = lossFunction(nn,dataset);
    
    // calculate the first gradient
    for(int i=0; i<NPARAMETERS; i++) {
        nnh = *nn;
        nnh.par[i] = nn->par[i] + h;
        grad[i] = (lossFunction(&nnh,dataset) - nn->strct.lf)/h;
    }
    
    // optimize the cost function
    while(nn->strct.lf > accuracy) {
    //do {

        // compute the search direction
        for (int i=0; i<NPARAMETERS; i++) {
            p[i] = 0.0;
            for (int j=0; j<NPARAMETERS; j++) {
                p[i] += -hess[i][j]*grad[j];
            }
        }
        
        // normalize the search direction
        double nrm = norm(p);
        for (int i=0; i<NPARAMETERS; i++) {
            p[i] = p[i]/nrm;
        }
        
        #ifdef LINESEARCH
            double alpha = linesearch(nn,grad,p,dataset,learningrate);
        #else
            double alpha = learningrate;
        #endif
        
        // calculate new parameters
        #ifdef PARALLEL
            #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(grad,grad2,alpha,p) shared(nn,y,s)
        #endif
        for(int i=0; i<NPARAMETERS; i++) {
            s[i] = alpha*p[i];
            nn->par[i] = nn->par[i] + s[i];
        }
        //if(norm(s)<1e-7) break;
        
        // calculate the gradient
        #ifdef PARALLEL
            #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(h,dataset,nn,nnh) shared(grad,grad2) private(temp)
        #endif
        for(int i=0; i<NPARAMETERS; i++) {
            nnh = *nn;
            nnh.par[i] = nn->par[i] + h;
            grad2[i] = grad[i];
            grad[i] = (lossFunction(&nnh,dataset) - nn->strct.lf)/h;
            y[i] = grad[i] - grad2[i];
        }
        
        // check positive finiteness (remember, p = - H*gradf)
        /*if(dot(grad,p)>0) {
            double nrm = norm(grad);
            for (int i=0; i<NPARAMETERS; i++) {
                p[i] = -grad[i]/nrm;
                for (int j=0; j<NPARAMETERS; j++) {
                    hess[i][j] = 0.0;
                }
                hess[i][i] = 1.0;
            }
        }*/
        // check the curvature condition
        if(dot(y,s)>0) {
            double nrm = norm(grad);
            for (int i=0; i<NPARAMETERS; i++) {
                p[i] = -grad[i]/nrm;
                for (int j=0; j<NPARAMETERS; j++) {
                    hess[i][j] = 0.0;
                }
                //hess[i][i] = 1.0;
                hess[i][i] = dot(y,s)/dot(y,y);
            }
        }

        double rho = dot(s,y);
        if (fabs(rho) <= 0.0) rho = 1e3;
        rho = 1./rho;
        
        double yprime[NPARAMETERS];
        for (int i=0; i<NPARAMETERS; i++) {
            yprime[i] = 0.0;
            for (int j=0; j<NPARAMETERS; j++) {
                yprime[i] += hess[i][j]*y[j];
            }
        }
        double sig = dot(y,yprime);
    
        for (int i=0; i<NPARAMETERS; i++) {
            for (int j=i; j<NPARAMETERS; j++) {
                hess[i][j] = hess[i][j] + s[i]*s[j]*rho*(rho*sig + 1.0) - (yprime[i]*s[j] + s[i]*yprime[j])*rho;
                hess[j][i] = hess[i][j];
            }
        }
    
        nn->strct.lf = lossFunction(nn,dataset);
        
        printf("alpha = %f, lf = %f, gradf = %f\n",alpha,nn->strct.lf,norm(grad));
        
        snapshot(nn); // TODO: delete
        
    } //while(norm(grad) > accuracy);
}

void snapshot(union NN *nn) {
    FILE *f1 = fopen("nn.dat", "w");
    if(f1 == NULL){
        printf("Error opening file!\n");
        exit(1);
    }

    for(int i = 0; i<NINPUTS; i++) {
        fprintf(f1,"w%i %f\n",i,nn->strct.iweight[i]);
    }
    for(int i = 0; i<NINPUTS; i++) {
        fprintf(f1,"t%i %f\n",i,nn->strct.itheta[i]);
    }
    for (int n = 0; n<NNEURONS; n++) {
        for(int i = 0; i<NINPUTS; i++) {
            fprintf(f1,"w%i%i %f\n",n,i,nn->strct.nweight[n][i]);
        }
    }
    for (int n = 0; n<NNEURONS; n++) {
        fprintf(f1,"t%i %f\n",n,nn->strct.ntheta[n]);
    }
    for (int n = 0; n<NOUTPUTS; n++) {
        for(int i = 0; i<NNEURONS; i++) {
            fprintf(f1,"w%i%i %f\n",n,i,nn->strct.oweight[n][i]);
        }
    }
    for (int n = 0; n<NOUTPUTS; n++) {
        fprintf(f1,"t%i %f\n",n,nn->strct.otheta[n]);
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
            nn->strct.iweight[i] = val;
        }
    }
    for(int i = 0; i<NINPUTS; i++) {
        fscanf(f1, "%s", buff);
        strncpy(key1, buff, 255);
        fscanf(f1, "%s", buff);
        sscanf(buff, "%lf", &val);
        sprintf(key2, "%s%d", "t", i);
        if(strcmp(key1, key2)==0) {
            nn->strct.itheta[i] = val;
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
                nn->strct.nweight[n][i] = val;
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
            nn->strct.ntheta[n] = val;
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
                nn->strct.oweight[n][i] = val;
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
            nn->strct.otheta[n] = val;
        }
    }

    fclose(f1);
}