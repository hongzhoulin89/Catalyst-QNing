function [z,it,train_loss_list,dualgaplist,train_acc_list,test_acc_list, test_loss_list, nnzlist] = quickening(Xtrain,Ytrain,Xtest,Ytest,param, nb_it)

%   This function run catalyst to optimize the given objective function
%   Outputs:
%       z               : the last iterate, which is the prox of xk
%       it              : list of nb of epochs passed 
%       train_loss_list : list of values of f(z) 
%       dualgaplist     : list of Fenchel duality gap at each z
%       train_acc_list  : list of training accuracy at each z
%       test_acc_list   : list of testing accuracy at each z
%       test_loss_list  : list of loss functions on test set at each z 
%       nnzlist         : list of number of non zero components of z

ntrain=size(Xtrain,2);
p=size(Xtrain,1);
ntest = size(Xtest,2);

%%%% Set parameters for Test evaluation
param_test = param;
param_test.mu = 0;
param_test.lambda = 0;

%%%% Initialization
xk = zeros(p,1);
train_loss= compute_loss(xk,Ytrain,Xtrain,param);
train_loss_list = [train_loss];
it = [0];

%%%% Evaluate Train and Test accuracy
Ytrain_pred = linear_prediction(Xtrain,xk);
train_acc = sum(Ytrain_pred == Ytrain)/ntrain;

Ytest_pred = linear_prediction(Xtest,xk);
if ntest == 0
    test_acc= 0;
    test_loss= 0;
else
    test_acc = sum(Ytest_pred == Ytest)/ntest;
    test_loss = compute_loss(xk,Ytest,Xtest,param_test);
end
train_acc_list = [train_acc];
test_acc_list = [test_acc];
test_loss_list = [test_loss];
fprintf('Iteration 0, Train loss: %g, Train accuracy %g, Test accuracy: %g, Test loss: %g \n',train_loss, train_acc, test_acc,test_loss);

%%%% Evaluate Duality gap
dualgap = compute_dualgap(xk,Ytrain,Xtrain,param); % Fenchel duality gap
dualgap_init = min(dualgap,train_loss);
fprintf('Inital duality gap :%g \n',dualgap_init);

dualgaplist = [dualgap];
nnzlist = [nnz(xk)];
   
%%%%% Initialization
s_list = [];
y_list = [];
rho_list = [];
restart = 0;
xk_past = xk;

%[g,F,z] = approxgrad_qn_svrg(w,Y,X,param);
[g, F, z, total_it] = approxgrad(xk,Ytrain,Xtrain,param,xk,xk_past);

d = g/param.kappa; % d = H*g
kappa = param.kappa;
ii = 0 ;

while total_it < nb_it
    ii = ii+1;
    x_test = xk - d;
    if ~param.nonsc
        [g_test, F_test,z_test, nb_grad] = approxgrad(x_test,Ytrain,Xtrain,param,z,xk_past);
        total_it = total_it + nb_grad; %%% nb of grad evaluation
        if param.restart
            if F_test > F - 0.5*g'*g/kappa
                x_test_new = z;
                [g_test_new, F_test_new, z_test_new,nb_grad] = approxgrad(x_test_new,Ytrain,Xtrain,param,z,xk_past);
                total_it = total_it + nb_grad;
                restart =1;
%                 if param.lbfgs_type == 1
                    s = x_test_new -x_test;
                    y = g_test_new -g_test;
%                 else
%                     s = x_test_new -xk;
%                     y = g_test_new -g;
%                 end
                x_test = x_test_new;
                g_test = g_test_new;
                F_test = F_test_new;
                z_test = z_test_new;
            else
                s = x_test -xk;
                y = g_test -g;
            end
        end
    else
        if param.restart
            if compute_loss(x_test,Ytrain,Xtrain,param) > F - 0.5*g'*g/kappa
                x_test = z;
                restart = 1;
            else
                restart = 0;
            end
            [g_test, F_test, z_test,nb_grad] = approxgrad(x_test,Ytrain,Xtrain,param,z,xk_past);
            total_it = total_it + nb_grad;
        end
        s = x_test -xk;
        y = g_test -g;
    end

    %%%% Compute Hk*gk
    [d, s_list, y_list, rho_list]= compute_direction(s_list,y_list,rho_list,s,y,g_test,param);

    %%%% Renew variables
    xk_past = xk;
    xk = x_test;
    z = z_test;
    g = g_test;
    F = F_test;

    dualgap = compute_dualgap(z,Ytrain,Xtrain,param);

    %%%% Save values
    value = F - 0.5*g'*g/kappa;
    fprintf('Iter %d, restart: %d, loss: %g,dualgap: %g \n',ii,restart, value,dualgap);
    restart = 0;
    train_loss_list = [train_loss_list, value];    
    it = [it, total_it];
    nnzlist = [nnzlist, nnz(z)/size(z,1)];
    dualgaplist = [dualgaplist, dualgap];
    
    
    %%%% Train accuracy 
    Ytrain_pred = linear_prediction(Xtrain,z);
    train_acc = sum(Ytrain_pred == Ytrain)/ntrain;

    %%%% Test accuracy
    Ytest_pred = linear_prediction(Xtest,z);
    if ntest == 0
        test_acc= 0;
        test_loss = 0; 
    else
        test_acc = sum(Ytest_pred == Ytest)/ntest;
        test_loss = compute_loss(z,Ytest,Xtest,param_test);
    end
    
    fprintf('Train accuracy %g, Test accuracy: %g, Test loss: %g \n', train_acc, test_acc,test_loss);
    
    train_acc_list = [train_acc_list, train_acc];
    test_acc_list = [test_acc_list, test_acc];
    test_loss_list = [test_loss_list,test_loss ];
end
    
end