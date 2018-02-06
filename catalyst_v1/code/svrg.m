function [w,it,train_loss_list,dualgaplist,train_acc_list,test_acc_list,test_loss_list,nnzlist] = svrg(Xtrain,Ytrain,Xtest,Ytest,param,nb_it)

%   This function run svrg to optimize the given objective function
%   Outputs:
%       w               : the last iterate
%       it              : list of nb of epochs passed 
%       train_loss_list : list of values of f(w) 
%       dualgaplist     : list of Fenchel duality gap at each w
%       train_acc_list  : list of training accuracy at each w
%       test_acc_list   : list of testing accuracy at each w
%       test_loss_list  : list of loss functions on test set at each w 
%       nnzlist         : list of number of non zero components of w

[p,ntrain]=size(Xtrain);
ntest = size(Xtest,2);

% Set test set parameters
param_test = param;
param_test.mu = 0;
param_test.lambda = 0;

%%%%% Initialization
w = zeros(p,1);
train_loss = compute_loss(w,Ytrain,Xtrain,param);
train_loss_list = [train_loss];
it = [0];
nnzlist=[0];

count =0;

%%%% Train and Test accuracy
Ytrain_pred = linear_prediction(Xtrain,w);
train_acc = sum(Ytrain_pred == Ytrain)/ntrain;

Ytest_pred = linear_prediction(Xtest,w);
if ntest == 0
    test_acc= 0;
    test_loss= 0;
else
    test_acc = sum(Ytest_pred == Ytest)/ntest;
    test_loss = compute_loss(w,Ytest,Xtest,param_test);
end
train_acc_list = [train_acc];
test_acc_list = [test_acc];
test_loss_list = [test_loss];
fprintf('Iteration 0, Train loss: %g, Train accuracy %g, Test accuracy: %g, Test loss: %g \n', train_loss, train_acc, test_acc,test_loss);

%%%% Evaluate Duality gap
dualgap = compute_dualgap(w,Ytrain,Xtrain,param); % Fenchel duality gap
dualgap = min(dualgap,train_loss);
fprintf('Inital duality gap :%g \n',dualgap);
dualgaplist = [dualgap];

for ii=1:nb_it
    if strcmp(param.model,'lasso')
        label = 0;
        [w, nb_grad, F]=...
        mex_svrg_elasticnet_onepass(Ytrain,Xtrain,label,param.Lips,param.lambda,param.mu,w,param.m,param.eta);
    elseif strcmp(param.model,'elasticnet')
        label = 0;
        [w, nb_grad, F]=...
        mex_svrg_elasticnet_onepass(Ytrain,Xtrain,label,param.Lips,param.lambda,param.mu,w,param.m,param.eta);
    elseif strcmp(param.model,'logi')
        label = 1;
        [w, nb_grad,F]=...
        mex_svrg_smooth_onepass(Ytrain,Xtrain,label,param.Lips,param.mu,w,param.m,param.eta); 
    end
    count = count + nb_grad;
    %%%% Save values
    train_loss = F;
    if param.mu>0 || param.lambda>0
        dualgap = compute_dualgap(w,Ytrain,Xtrain,param);
        dualgaplist= [dualgaplist, dualgap];
        fprintf('Iter %d, loss: %g, dualgap: %g \n',ii, train_loss, dualgap);
    else
        fprintf('Iter %d, loss: %g \n',ii, train_loss);
    end

    %%%% Train accuracy 
    Ytrain_pred = linear_prediction(Xtrain,w);
    train_acc = (sum(Ytrain_pred == Ytrain)/ntrain);

    %%%% Test accuracy
    Ytest_pred = linear_prediction(Xtest,w);
    if ntest == 0
        test_acc= 0;
        test_loss = 0; 
    else
        test_acc = sum(Ytest_pred == Ytest)/ntest;
        test_loss = compute_loss(w,Ytest,Xtest,param_test);
    end
    
    fprintf('Train accuracy %g, Test accuracy: %g, Test loss: %g \n', train_acc, test_acc,test_loss);

    
    train_loss_list = [train_loss_list, train_loss];    
    it = [it, count];
    nnzlist = [nnzlist, nnz(w)];
    train_acc_list = [train_acc_list, train_acc];
    test_acc_list = [test_acc_list, test_acc];
    test_loss_list = [test_loss_list,test_loss ];
end
    
end