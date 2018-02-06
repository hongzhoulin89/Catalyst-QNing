# Catalyst-QNing

This is a Matlab code of **Catalyst/QNing**, which are accelerated algorithms for first-order optimization methods. 

The function 'code/example.m' gives an example of running Catalyst and QNing with SVRG algorithm. 
To run the example, download the folder **catalyst_v1** and in **Matlab** type:

```
>> cd catalyst_v1/code      % Change to the code directory
>> mexAll                   % Compile mex files
>> example                  % Run Catalyst/QNing SVRG to minimize logistic regression 
```

The default parameters are set as in theory, see the file 'example' for more details.  
