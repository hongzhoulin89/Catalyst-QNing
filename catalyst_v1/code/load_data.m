function [Xtrain,Ytrain,Xtest,Ytest] = load_data(dataset)

%   This function load training and testing data 
%   (the testing data is optional)

%%%%%%%%%%%%%%%%  Loading training %%%%%%%%%%%%%%%%%%%
datapath='../data/';

trainfname=[datapath dataset '_train.mat'];
testfname=[datapath dataset '_test.mat'];
load(trainfname,'Xtrain','Ytrain'); % Xtrain is a p*ntrain matrix, Ytrain is a ntrain*1 vector
p=size(Xtrain,1);
ntrain=size(Xtrain,2);
if ntrain ~= size(Ytrain,1);
    error('Size of Xtrain and Ytrain must agree. \n');
end

fprintf('Training data loaded, size: %d, dimension: %d \n',ntrain,p);

%%%%%%%%%%%%%%%%  Optional: Testing %%%%%%%%%%%%%%%%%%%
if ~exist(testfname)
    Xtest = [];
    Ytest = []; 
    fprintf('No test set \n');
else
    load(testfname,'Xtest','Ytest'); % Xtest is a p*ntest matrix, Ytest is a ntest*1 vector
    ntest=size(Xtest,2);  
    if ntest ~= size(Ytest,1);
        error('Size of Xtest and Ytest must agree. \n');
    end
    fprintf('Testing data loaded, size: %d \n',ntest);
end

end
