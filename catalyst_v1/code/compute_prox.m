function[z0] = compute_prox(w0,Ytrain,Xtrain,param,yk)
%     Compute the proximal operator of inner loop function
%       z0 = prox(w0 - eta*grad)
%       where 
%           eta = 1/(Lips+kappa+mu)
%           grad = grad_square_loss(w0) + mu*w0 + kappa*(w0-yk)   
%
%     Outputs:  
%       z0      prox vector
 

n =size(Ytrain,1);
z0 = wthresh(w0 - (Xtrain*(Xtrain'*w0-Ytrain)/n+param.kappa*(w0-yk)+param.mu*w0)/(param.Lips+param.kappa+param.mu),'s',param.lambda/(param.Lips+param.kappa+param.mu));

end