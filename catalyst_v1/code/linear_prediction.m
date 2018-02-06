function [y] = linear_prediction(X,x)
%     Evaluate the prediction of the linear classification
%       Output:
%           y   classification vector of 1 or -1 

    y = ((x'*X) >0)';
    y = (y-0.5)*2;
end