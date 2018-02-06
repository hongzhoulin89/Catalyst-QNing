function[dualgap] = compute_dualgap(w,Y,X,param)

%   Compute the Fenchel duality gap at point w
%   Output :
%       dualgap  Fenchel duality gap of the objective function

n= size(Y,1);

if strcmp(param.model,'lasso')
    grad = (X'*w-Y)/n;
    dualgap = 0.5*n*grad'*grad;
    grad = min(1,param.lambda/max(abs(X*grad)))*grad;
    dualgap = dualgap + param.lambda*sum(abs(w))+0.5*n*grad'*grad + grad'*Y;
elseif strcmp(param.model,'elasticnet')
    lambda = param.lambda;
    grad = (X'*w-Y)/n;
    dualgap = n*grad'*grad + 0.5*param.mu*w'*w + lambda*sum(abs(w))+grad'*Y;
    res = (abs(-X*grad) - lambda);
    dualgap = dualgap+ 0.5*sum((0.5*(res + abs(res))).^2)/param.mu;
elseif strcmp(param.model,'logi')
    grad = -(Y./(1+exp(Y.*(X'*w))) );
    z = -X*grad/n;
    dualgap = sum(log(1+exp(-Y.*(X'*w))))/n + 0.5*param.mu*w'*w + sum((1+grad./Y).*log(1+Y.*grad) - (grad./Y).*log(-grad./Y))/n + 0.5*z'*z/param.mu;
end

end