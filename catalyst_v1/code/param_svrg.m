function [param] = param_svrg(Xtrain, model, mu, lambda, default)

%   This function initialize the parameters for catalyst-svrg
%
%   Outputs:  
%       param: a field contains following parameters
%           model  : 'logi', 'elasticnet' or 'lasso'
%           mu     :  l2 parameter
%           lambda :  l1 paramater
%           Lips   :  Lipschitz constant for the incremental gradient
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

if strcmp(model,'logi')
    param.mu = mu/ntrain;
    param.lambda =0;
elseif strcmp(model,'elasticnet')
    param.mu = mu/ntrain;
    param.lambda = lambda/ntrain;
elseif strcmp(model,'lasso')
    param.mu = 0;
    param.lambda = lambda/ntrain;
end

%%%%%%%%%%%%%%%%%%%%%  PART 1: DEFAULT Parameters %%%%%%%%%%%%%%%%%%%%%%%%%
if default
    if strcmp(model,'logi')
        param.Lips = max_Lips/4;
    elseif strcmp(model,'elasticnet') || strcmp(model,'lasso')
        param.Lips = max_Lips;
    end
    
    param.m  = ntrain;  % Nb epochs before reseting the full gradient in SVRG
    param.eta = 1/( param.Lips+param.mu); % Stepsize for SVRG in inner loop

else       
%%%%%%%%%%%%%%%%%%%%   PART 2: PERSONALIZED Parameters   %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%   Specify all parameters in PART 1  %%%%%%%%%%%%%%%%%%%%

%     param.Lips = ;
%     param.m  =   ;  % Nb epochs before reseting the full gradient in SVRG
%     param.eta =  ; % Stepsize for SVRG in inner loop
    



end





end