/*  NORMALIZECOLS.C  Normalize the columns of a matrix
 Syntax:      B = normalizecols(A)
 or B = normalizecols(A,p)
 The columns of matrix A are normalized so that norm(B(:,n),p) = 1. */
#include <math.h>
#include <algorithm>
#include <limits>
#include <stdlib.h>
#include "mex.h"

#define IS_REAL_2D_FULL_DOUBLE(P) (!mxIsComplex(P) && \
mxGetNumberOfDimensions(P) == 2 && !mxIsSparse(P) && mxIsDouble(P))
#define IS_REAL_SCALAR(P) (IS_REAL_2D_FULL_DOUBLE(P) && mxGetNumberOfElements(P) == 1)

/*extern "C" mxArray *mxCreateSharedDataCopy(const mxArray *pr);*/

#if defined(NAN_EQUALS_ZERO)
#define IsNonZero(d) ((d)!=0.0 || mxIsNaN(d))
#else
#define IsNonZero(d) ((d)!=0.0)
#endif

double der(int label, double y, double x)
{
    double out_value;
    if (label == 0) /* Ridge */
        out_value = x-y;
    else if (label == 1) /* Logistic */
        out_value =  -y/(1.0 + exp(x*y));
    return out_value;
}

double logexp(double x) {
    if (x < -30) {
        return 0;
    } else if (x < 30) {
        return log(1.0 + exp( x ) );
    } else {
        return x;
    }
};


double loss(int label, double y, double x)
{
    double out_value;
    if (label == 0) /* Ridge */
        out_value = 0.5*(x-y)*(x-y);
    else if (label == 1) /* Logistic */
        out_value = logexp(-y*x);
    return out_value;
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Macros for the input arguments */
    #define Y_IN prhs[0]
    #define X_IN prhs[1]
    #define Label_IN prhs[2]
    #define Lips_IN prhs[3]
    #define Lambda_IN prhs[4]
    #define W_IN prhs[5]
    #define M_IN prhs[6]
    #define Theta_IN prhs[7]
    #define Kappa_IN prhs[8]
    #define Y_k_IN prhs[9]
    #define Delta_IN prhs[10]
    
    
    /* Define output arguments*/
    mxArray *Wreturn, *Nb_itreturn,*Valuelistreturn;
    mwIndex *jc,*ir;
    
    mxArray *X_tilde, *V_tilde,*X_avg;
    double *x_tilde, *v_tilde, *x_avg;
    /* Define variables*/
    double *Y, *X, *W, *Nb_it,*Y_k, L, Lips, lambda, theta, z, value,deriv,loss_value,delta,dualgap,update_parameter,deriv_tilde,kappa,deriv_norm,para1;
    int n, p,jj, ii, kk, tt, nb_it, Label, i_nb,m,sparse=0;
    
    /* Define output variables */
    double *W_out,*Valuelist_out;
    
    if(nrhs < 3 || nrhs > 11) /* Check the number of arguments */
        mexErrMsgTxt("Wrong number of input arguments.");
    else if(nlhs > 3)
        mexErrMsgTxt("Too many output arguments.");
    
    if(!IS_REAL_2D_FULL_DOUBLE(Y_IN)) /* Check Y */
        mexErrMsgTxt("Y must be a real 2D full double array.");
    else Y = mxGetPr(Y_IN); /* Get the pointer to the data of Y */
    
    if (mxIsSparse(X_IN))
    {
        sparse = 1;
        jc = mxGetJc(X_IN);
        ir = mxGetIr(X_IN);
        p = mxGetM(X_IN); /* Get number of rows */
        n = mxGetN(X_IN); /* Get number of columns */
        X = mxGetPr(X_IN); /* Get the pointer to the data of X */
    }
    else if(!IS_REAL_2D_FULL_DOUBLE(X_IN)) /* Check X */
        mexErrMsgTxt("X must be a real 2D full double array.");
    else
    {
        p = mxGetM(X_IN); /* Get number of rows */
        n = mxGetN(X_IN); /* Get number of columns */
        X = mxGetPr(X_IN); /* Get the pointer to the data of X */
    }
    
    
    if(!IS_REAL_SCALAR(Label_IN))
        mexErrMsgTxt("Label must be a real double scalar.");
    else
        Label = (int) mxGetScalar(Label_IN); /* Get Label: 0=ridge, 1=logit */
    
    
    if(!IS_REAL_SCALAR(Lips_IN))
        mexErrMsgTxt("Lips must be a real double scalar.");
    else
        Lips = mxGetScalar(Lips_IN); /* Get Lips */
    
    
    if(nrhs < 5) /* If lambda is unspecified, set it to a default value */
        lambda = 0.0;
    else /* If Lambda was specified, check that it is a real double scalar */
        if(!IS_REAL_SCALAR(Lambda_IN))
            mexErrMsgTxt("lambda must be a real double scalar.");
        else /* Get lambda */
            lambda = mxGetScalar(Lambda_IN);
    
    if(nrhs < 6) /* If w is unspecified, set it to a default value */
    {
        Wreturn = mxCreateDoubleMatrix(p, 1, mxREAL);
        W_out = mxGetPr(Wreturn);
        for(ii = 0; ii< p; ii++)
        {
            W_out[ii] = 0.0;
        }
    }
    else /* If W was specified, check that it is a real double scalar */
        if(!IS_REAL_2D_FULL_DOUBLE(W_IN))
            mexErrMsgTxt("W must be a real 2D full double array.");
        else /* Get W */
        {
            W = mxGetPr(W_IN);
            Wreturn = mxCreateDoubleMatrix(p, 1, mxREAL);
            W_out = mxGetPr(Wreturn);
            for(ii = 0; ii< p; ii++)
            {
                W_out[ii] = W[ii];
            }
        }
    

    /* Initialization of m */
    if(nrhs < 7) /* If Alpha is unspecified, set it to a default value */
    {
        m = 2*n;
    }
    else if(!IS_REAL_SCALAR(M_IN))
        mexErrMsgTxt("m must be a int scalar.");
    else // Get m
    {
        m = mxGetScalar(M_IN);
        //printf("m = %d ends \n", m);
    }
    
    /* Initialization of theta */
    if(nrhs < 8) /* If Alpha is unspecified, set it to a default value */
    {
        theta = 0.1/Lips;
    }
    else if(!IS_REAL_SCALAR(Theta_IN))
        mexErrMsgTxt("m must be a int scalar.");
    else // Get m
    {
        theta = mxGetScalar(Theta_IN);
    }
    
    
    if(nrhs < 9) // If kappa is unspecified, set it to a default value
        kappa = 0.0;
    else if(!IS_REAL_SCALAR(Kappa_IN)) // If Kappa was specified, check that it is a real double scalar
        mexErrMsgTxt("kappa must be a real double scalar.");
    else // Get kappa
        kappa = mxGetScalar(Kappa_IN);
    
    
    /* Initialization of Y_k */
    if(nrhs < 10) // If y_k is unspecified, set it to a default value
    {
        Y_k = mxGetPr(mxCreateDoubleMatrix(p, 1, mxREAL));
        for(ii = 0; ii< p; ii++)
        {
            Y_k[ii] = 0.0;
        }
    }
    else if(!IS_REAL_2D_FULL_DOUBLE(Y_k_IN)) // If Y_k was specified, check that it is a real double scalar
        mexErrMsgTxt("y_k must be a real 2D full double array.");
    else // Get Y_k
    {
        Y_k = mxGetPr(Y_k_IN);
    }
    
    
    if(nrhs < 11) /* If Eps is unspecified, set it to a default value */
        delta = std::numeric_limits<double>::max();
    else /* If Kappa was specified, check that it is a real double scalar */
        if(!IS_REAL_SCALAR(Delta_IN))
            mexErrMsgTxt("delta must be a real double scalar.");
        else /* Get kappa */
        {
            delta = mxGetScalar(Delta_IN);
            printf("Inner loop starts: find an adaptive-solution with delta = %f  \n", delta);
        }
    
    
    L = Lips+lambda+kappa;
    // this is gamma in the paper of saga
    //update_parameter = 1.0/(1.0+theta*lambda);
    //dualgap = std::numeric_limits<double>::infinity();
    
    // Define Output list
    Valuelistreturn = mxCreateDoubleMatrix(1, 1, mxREAL);
    Valuelist_out = mxGetPr(Valuelistreturn);
    
    
    X_tilde = mxCreateDoubleMatrix(p, 1, mxREAL);
    x_tilde = mxGetPr(X_tilde);
    
    V_tilde = mxCreateDoubleMatrix(p, 1, mxREAL);
    v_tilde = mxGetPr(V_tilde);
    
    X_avg = mxCreateDoubleMatrix(p, 1, mxREAL);
    x_avg = mxGetPr(X_avg);
    
    // Define x_tilde
    for(ii = 0; ii< p; ii++)
    {
        x_tilde[ii] = W_out[ii];
    }
    
    
    // Define v_tilde
    
    for(ii = 0; ii< p; ii++)
    {
        v_tilde[ii] = (lambda+kappa)*x_tilde[ii]- kappa*Y_k[ii] ;
    }
    
    for (ii =0; ii < n; ii++)
    {
        if (sparse){
            z = 0.0;
            for(kk=jc[ii]; kk<jc[ii+1]; kk++)
            {
                z += x_tilde[ir[kk]]*X[kk];
            }
        }
        else{
            for (kk =0, z = 0.0; kk < p; kk++)
            {
                z += x_tilde[kk]*X[kk+ii*p];
            }
        }
        deriv = der(Label, Y[ii],z)/n;
        //printf("deriv = %f ends \n", deriv);
        
        if (sparse){
            for(kk=jc[ii]; kk<jc[ii+1]; kk++)
            {
                v_tilde[ir[kk]] += deriv*X[kk];
            }
        }
        else{
            for (kk =0; kk < p; kk++)
            {
                v_tilde[kk] += deriv*X[kk+ii*p];
            }
        }
    }
    
    
    
    double eps_norm;
    nb_it = 0;
    
    
    
    do{

        // Define x_avg
        for(ii = 0; ii< p; ii++)
        {
            x_avg[ii] = 0.0;
        }
        
        
        // Inner loop
        for(jj = 0; jj< m; jj++) /* Compute a matrix with normalized columns */
        {
            ii = rand()%n;
            
            /* Compute the inner product X(:,ii))'*W */
            if (sparse)
            {
                z = 0.0;
                for(kk=jc[ii]; kk<jc[ii+1]; kk++)
                {
                    z += W_out[ir[kk]]*X[kk];
                }
            }
            else{
                for (kk =0, z = 0.0; kk < p; kk++)
                {
                    z += W_out[kk]*X[kk+ii*p];
                }
            }
            
            /* Compute derivative and function value */
            deriv = der(Label, Y[ii],z);
            //loss_value = loss(Label, Y[ii],z);
            
            /* Compute the inner product X(:,ii))'*x_tilde */
            if (sparse)
            {
                z = 0.0;
                for(kk=jc[ii]; kk<jc[ii+1]; kk++)
                {
                    z += x_tilde[ir[kk]]*X[kk];
                }
            }
            else{
                for (kk =0, z = 0.0; kk < p; kk++)
                {
                    z += x_tilde[kk]*X[kk+ii*p];
                }
            }
            deriv_tilde = der(Label, Y[ii],z);
            
            
            /* Update W */
            if (sparse)
            {
                for (kk =0; kk < p; kk++)
                {
                    W_out[kk] = W_out[kk] -theta*((lambda+kappa)*(W_out[kk]-x_tilde[kk]) + v_tilde[kk]);
                }
                for(kk=jc[ii]; kk<jc[ii+1]; kk++)
                {
                    W_out[ir[kk]] += -theta*(deriv-deriv_tilde)*X[kk];
                }
                
            }
            else{
                for (kk =0; kk < p; kk++)
                {
                    W_out[kk] = W_out[kk] -theta*( (deriv-deriv_tilde)*X[kk+ii*p]+(lambda+kappa)*(W_out[kk]-x_tilde[kk]) + v_tilde[kk]);
                }
            }
            
            // Update x_avg
            for (kk =0; kk < p; kk++)
            {
                x_avg[kk] =  ((double)jj)/(jj+1.0)*x_avg[kk] + 1.0/(jj+1.0)*W_out[kk] ;
            }
        }
        

        
        
        // Define W_out
        for(ii = 0; ii< p; ii++)
        {
            W_out[ii] = x_avg[ii];
        }
        
        // Define x_tilde
        for(ii = 0; ii< p; ii++)
        {
            x_tilde[ii] = W_out[ii];
        }
        
        // Define v_tilde
        
        for(ii = 0; ii< p; ii++)
        {
            v_tilde[ii] = (lambda+kappa)*x_tilde[ii]- kappa*Y_k[ii] ;
        }
        
        for (ii =0; ii < n; ii++)
        {
            if (sparse){
                z = 0.0;
                for(kk=jc[ii]; kk<jc[ii+1]; kk++)
                {
                    z += x_tilde[ir[kk]]*X[kk];
                }
            }
            else{
                for (kk =0, z = 0.0; kk < p; kk++)
                {
                    z += x_tilde[kk]*X[kk+ii*p];
                }
            }
            
            deriv = der(Label, Y[ii],z)/n;
            //printf("deriv = %f ends \n", deriv);
            if (sparse){
                for(kk=jc[ii]; kk<jc[ii+1]; kk++)
                {
                    v_tilde[ir[kk]] += deriv*X[kk];
                }
            }
            else{
                for (kk =0; kk < p; kk++)
                {
                    v_tilde[kk] += deriv*X[kk+ii*p];
                }
            }
        }
        
        deriv_norm = 0.0;
        for (kk =0; kk < p; kk++)
        {
            deriv_norm += v_tilde[kk]*v_tilde[kk];
        }
        deriv_norm = deriv_norm/( 2.0*(lambda+kappa));
        nb_it++;
        
        
        // Compute eps_norm
        eps_norm = 0.0;

        for (kk =0; kk < p; kk++)
        {
            eps_norm += (W_out[kk]-Y_k[kk])*(W_out[kk]-Y_k[kk]);
        }
        eps_norm = eps_norm*0.5*kappa*delta;
        
        //printf("eps_norm = %e \n",eps_norm);
        printf("Inner loop iteration %d:  subproblem's dualgap = %e, adaptive eps = %e \n", nb_it,deriv_norm,eps_norm);     
        
    }while(deriv_norm > eps_norm + pow (10.0, -11.0));
    printf("Inner loop ends after %d epochs \n", nb_it);
    
    /* Evaluate the function value*/
    value = 0.0;
    
    for (ii =0; ii < n; ii++)
    {
        if (sparse)
        {
            z = 0.0;
            for(kk=jc[ii]; kk<jc[ii+1]; kk++)
            {
                z += W_out[ir[kk]]*X[kk];
            }
        }
        else{
            for (kk =0, z = 0.0; kk < p; kk++)
            {
                z += W_out[kk]*X[kk+ii*p];
            }
        }
        value += loss(Label, Y[ii],z);
    }
    value = value/n;
    
    /* Compute the l2 norm of W_prox + l2 norm */
    for (kk =0, z = 0.0; kk < p; kk++)
    {
        z += 0.5*lambda*W_out[kk]*W_out[kk]+ 0.5*kappa*(W_out[kk]-Y_k[kk])*(W_out[kk]-Y_k[kk]);
    }
    value += z;
    
    Valuelist_out[0] = value;
    
    Nb_itreturn = mxCreateDoubleMatrix(1, 1, mxREAL);
    Nb_it = mxGetPr(Nb_itreturn);
    Nb_it[0] = ((double) nb_it*m)/n;
    
    plhs[0] = Wreturn;
    plhs[1] = Nb_itreturn;
    //plhs[2] = Alphareturn;
    //plhs[2] = Betareturn;
    plhs[2] = Valuelistreturn;
    
    return;
}



