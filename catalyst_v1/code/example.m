%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                 Example of Catalyst/QuickeNing-SVRG                     %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   Part 1:  load and prepare data                                        %
%   Part 2:  apply SVRG and Catalyst-SVRG                                 %
%   Part 3:  plot comparison                                              %
%                                                                         %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%   Loading training and testing data    %%%%%%%%%%%%%%%%%%%
%                                                                         %
% DETAILS SEE load_data.m                                                 %
%                                                                         %
% TO USE YOUR OWN DATASET:                                                %
% a) Place the training data in: v1/data/NAME_train.mat                   %
% b) Set the variables names as                                           %
%       Xtrain : p*n matrix (p : dimension of feature, n: size of dataset)%
%       Ytrain : n*1 matrix                                               %
% c) (OPTIONAL) Place the testing data in: v1/data/NAME_test.mat          %
% d) Set the variables names as                                           %
%       Xtest : p*ntest matrix (ntest: size of testing data)              %
%       Ytest : ntest*1 matrix                                            %
%                                                                         %
% IMPORTANT MESSAGE:                                                      %
%       1) X, Y must be arrays of doubles                                 %
%       2) NORMALIZE YOUR DATASET                                         %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataset = 'covtype';     % NAME of dataset
[Xtrain,Ytrain,Xtest,Ytest] = load_data(dataset);


%%%%%%%%%%%%%%%%%%%%   Specify the LOSS FUNCTION  %%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
% Available loss functions:                                               %
% a) 'logi' : logistic loss + l2 regularization                           %
%       l2 parameter: mu/n                                                %
% b) 'elasticnet': square loss + l2 regularization + l1 regularization    %
%       l2 parameter: mu/n                                                %
%       l1 parameter: lambda/n                                            %
% c) 'lasso': square loss + l1 regularization                             %
%       l1 parameter: lambda/n                                            %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model = 'logi';     % Models: 'logi', 'elasticnet' or 'lasso'

mu = 0.01;          % relative l2 parameter
                    % default: mu = 0.1 or 0.01

lambda = 0;         % relative l1 parameter,
                    % default lambda =10 






%%%%%%%%%%%%%%%%% Set Optimization parameters for Catalyst %%%%%%%%%%%%%%%%
%                                                                         %
% DETAILS SEE param_catalyst.m                                            %
%                                                                         %
% default_catalyst = 1 -> USE DEFAULT PARAMETERS                          %
%                                                                         %
% TO personalize the parameters                                           %
% Step 1: Set default_catalyst =0                                         %
% Step 2: Specify the parameters in PART 2 of param_catalyst.m            %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

default_catalyst = 1;   % Use default parameter setting 
algo = 'svrg';          % Algo in the inner loop of Catalyst
                        % For the current version, only svrg is available

[param] = param_catalyst(Xtrain, model, mu, lambda, algo, default_catalyst);

nb_it = 50;            % Nb of epochs


%%%%%%%%%%%%%%%%%%%%%%%%%%  RUN Catalyst-SVRG   %%%%%%%%%%%%%%%%%%%%%%%%%%%
[w_catalyst,it_catalyst,train_loss_list_catalyst,dualgaplist_catalyst, train_acc_list_catalyst, test_acc_list_catalyst, test_loss_list_catalyst, nnzlist_catalyst]...
    = catalyst(Xtrain,Ytrain,Xtest,Ytest,param, nb_it);

% Save the results
filename_catalyst = sprintf('../output/%s/%s_%s_catalyst_%s_%s_warm_start=%g_kappa=%g_mu=%0.1e_lambda=%0.1e_nb_it=%d.mat',...
        algo,dataset, model, algo,param.stop_criterion, param.warm_start, param.kappa, param.mu, param.lambda,nb_it);
save(filename_catalyst,'it_catalyst','train_loss_list_catalyst','dualgaplist_catalyst', 'train_acc_list_catalyst', 'test_acc_list_catalyst', 'test_loss_list_catalyst','nnzlist_catalyst');




%%%%%%%%%%%%%%%%% Set Optimization parameters for SVRG %%%%%%%%%%%%%%%%%%%%
%                                                                         %
% DETAILS SEE param_svrg.m                                                %
%                                                                         %
% default_svrg = 1 -> USE DEFAULT PARAMETERS                              %
%                                                                         %
% TO personalize the parameters                                           %
% Step 1: Set default_svrg =0                                             %
% Step 2: Specify the parameters in PART 2 of param_svrg.m                %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

default_svrg = 1;           % Use default parameter setting 
[param2] = param_svrg(Xtrain, model, mu, lambda, default_svrg);

%%%%%%%%%%%%%%%%%%%%%%%%%%      RUN SVRG     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[w,it,train_loss_list,dualgaplist, train_acc_list, test_acc_list, test_loss_list, nnzlist] = svrg(Xtrain,Ytrain,Xtest,Ytest,param2, nb_it);

% Save the results
filename_svrg = sprintf('../output/%s/%s_%s_%s_mu=%0.1e_lambda=%0.1e_nb_it=%d.mat',...
        algo,dataset, model, algo, param2.mu, param2.lambda,nb_it);
save(filename_svrg,'it','train_loss_list','dualgaplist', 'train_acc_list', 'test_acc_list', 'test_loss_list','nnzlist');



%%%%%%%%%%%%%%%% Set Optimization parameters for QuickeNing %%%%%%%%%%%%%%%
%                                                                         %
% DETAILS SEE param_quickening.m                                          %
%                                                                         %
% default_quickening = 1 -> USE DEFAULT PARAMETERS                        %
%                                                                         %
% TO personalize the parameters                                           %
% Step 1: Set default_quickening =0                                       %
% Step 2: Specify the parameters in PART 2 of param_quickening.m          %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

default_quickening = 1;     % Use default parameter setting 
algo = 'svrg';              % Algo in the inner loop of Catalyst
                            % For the current version, only svrg is
                            % available

[param3] = param_quickening(Xtrain, model, mu, lambda, algo, default_quickening);


%%%%%%%%%%%%%%%%%%%%%%%%%%  RUN Catalyst-SVRG   %%%%%%%%%%%%%%%%%%%%%%%%%%%
[w_qning,it_qning,train_loss_list_qning,dualgaplist_qning, train_acc_list_qning, test_acc_list_qning, test_loss_list_qning, nnzlist_qning]...
    = quickening(Xtrain,Ytrain,Xtest,Ytest,param3, nb_it);

% Save the results
filename_quickening = sprintf('../output/%s/%s_%s_quickening_%s_%s_warm_start=%g_kappa=%g_mu=%0.1e_lambda=%0.1e_nb_it=%d.mat',...
        algo,dataset, model, algo,param3.stop_criterion, param3.warm_start, param3.kappa, param3.mu, param3.lambda,nb_it);
save(filename_quickening,'it_qning','train_loss_list_qning','dualgaplist_qning', 'train_acc_list_qning', 'test_acc_list_qning', 'test_loss_list_qning','nnzlist_qning');


%%%%%%%%%%%%%%%%%%%%%  Draw figures of comparison   %%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   Figures are saved in the dirsctory v1/figures/svrg                    %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

draw_figures(dataset,model, mu, lambda,filename_catalyst,filename_svrg,filename_quickening)





