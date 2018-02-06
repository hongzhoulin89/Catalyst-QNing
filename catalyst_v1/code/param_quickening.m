function [param] = param_quickening(Xtrain, model , mu, lambda, algo, default)

%   This function initialize the parameters for catalyst-svrg
%
%   Outputs:  
%       param: a field contains following parameters
%           model  : 'logi', 'elasticnet' or 'lasso'
%           algo   : 'svrg' (the algorithm to run the subproblems, only 'svrg' is available for the current version)
%           mu     :  l2 parameter
%           lambda :  l1 paramater
%           Lips   :  Lipschitz constant for the incremental gradient
%           kappa  :  catalyst's parameter
%   stop_criterion :  'onepass', 'absolute' or relative' (stopping criterion for catalyst's inner loop)
%       warm_start :  0, 1, 2, 3 (different warm start strategy, 0 is proposed in NIPs paper and 1, 2, 3 corresponds to C1, C2, C3 in the new paper)
%           m      :  svrg's parameter, it's the nb of epochs before reseting the full gradient
%           eta    :  svrg's stepsize

%%%%%%%%%%%%%%  Check whether the data is normalized   %%%%%%%%%%%%%%%%%%%%
Lips_list = sum(Xtrain.^2) ;
max_Lips = full(max(Lips_list));
min_Lips = full(min(Lips_list));

if max_Lips > min_Lips + 10^(-8)
    fprintf('The Data is NOT normalized \n');
    fprintf('The largest norm is %d \n',max_Lips);
    fprintf('The smallest norm is %d \n',min_Lips);
end

ntrain = size(Xtrain,2);
param.model = model;
param.algo = algo;

if strcmp(model,'logi')
    param.mu = mu/ntrain;
    param.lambda =0;
    param.nonsc = 0;
elseif strcmp(model,'elasticnet')
    param.mu = mu/ntrain;
    param.lambda = lambda/ntrain;
    param.nonsc = 0;
elseif strcmp(model,'lasso')
    param.mu = 0;
    param.lambda = lambda/ntrain;
    param.nonsc = 1;
end

%%%%%%%%%%%%%%%%%%%%%  PART 1: DEFAULT Parameters %%%%%%%%%%%%%%%%%%%%%%%%%
if default
    if strcmp(model,'logi')
        param.Lips = max_Lips/4;
    elseif strcmp(model,'elasticnet') || strcmp(model,'lasso')
        param.Lips = max_Lips;
    end
    % Set Catalyst parameters
    if param.Lips/(ntrain+1) > param.mu
        param.kappa = (param.Lips/(ntrain+1)- param.mu); 
    else
        param.kappa = (param.Lips/(ntrain+1));
    end
    param.stop_criterion = 'onepass'; % Choice: 'onepass' or relative'
    param.warm_start = 2;   % Choice: 0,1,2,3
    param.restart = 1;
    param.delta = param.kappa/36;
    param.limit_mem = 10;
    param.lbfgs_type = 0;
    param.c1 = 0.5;
    param.c2 = 0.5;
    % Set Inner loop parameters
    if strcmp(algo,'svrg')
        param.m  = ntrain;  % Nb epochs before reseting the full gradient in SVRG
        param.eta = 1/( param.Lips+param.kappa+param.mu); % Stepsize for SVRG in inner loop
    end    
else       
%%%%%%%%%%%%%%%%%%%%   PART 2: PERSONALIZED Parameters   %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%   Specify all parameters in PART 1  %%%%%%%%%%%%%%%%%%%%

%     param.Lips = ;
%     param.kappa = ;
%     param.stop_criterion = ; % Choice: 'onepass', 'absolute' or relative'
%     param.warm_start = ;   % Choice: 0,1,2,3
%     if strcmp(algo,'svrg')
%         param.m  =   ;  % Nb epochs before reseting the full gradient in SVRG
%         param.eta =  ; % Stepsize for SVRG in inner loop
%     end        



end





end