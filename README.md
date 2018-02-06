# Catalyst-QNing

Matlab codes of the **Catalyst** and **QNing** generic acceleration schemes for first-order optimization.

The function ‘code/example.m’ gives an example of the Catalyst (resp. QNing) acceleration scheme applied to the SVRG optimization algorithm.

To run the example, download the folder **catalyst_v1** and in **Matlab** type:

```
>> cd catalyst_v1/code      % Change to the code directory
>> mexAll                   % Compile mex files
>> example                  % Run Catalyst/QNing SVRG to minimize logistic regression 
```

The default parameters are set as suggested by the theoretical analysis [1,2,3]; see ‘example’ for more details.

## Paper 
1. [A Universal Catalyst for First-Order Optimization](http://papers.nips.cc/paper/5928-a-universal-catalyst-for-first-order-optimization.pdf) H. Lin, J. Mairal, Z. Harchaoui

2. [A Generic Quasi-Newton Algorithm
for Faster Gradient-Based Optimization](https://arxiv.org/pdf/1610.00960.pdf) H. Lin, J. Mairal, Z. Harchaoui

3. [Catalyst Acceleration for First-order Convex Optimization:
from Theory to Practice](https://arxiv.org/pdf/1712.05654.pdf) H. Lin, J. Mairal, Z. Harchaoui
