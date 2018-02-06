function[loss] = compute_loss(w,Y,X,param)
%     Compute the loss function
%       loss = f(w)

n = size(Y,1);

if strcmp(param.model,'lasso')
    z = (X'*w-Y);
    loss = 0.5*z'*z/n+ param.lambda*sum(abs(w));
elseif strcmp(param.model,'elasticnet')
    z = (X'*w-Y);
    loss = 0.5*z'*z/n+ param.lambda*sum(abs(w))+ 0.5*param.mu*w'*w;
elseif strcmp(param.model,'logi')
    loss = sum(log(1+exp(-Y.*(X'*w))))/n + 0.5*param.mu*w'*w;
end

end