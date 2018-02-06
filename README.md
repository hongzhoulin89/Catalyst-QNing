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

#### References 
1. [A Universal Catalyst for First-Order Optimization](http://papers.nips.cc/paper/5928-a-universal-catalyst-for-first-order-optimization.pdf) H. Lin, J. Mairal, Z. Harchaoui

2. [A Generic Quasi-Newton Algorithm
for Faster Gradient-Based Optimization](https://arxiv.org/pdf/1610.00960.pdf) H. Lin, J. Mairal, Z. Harchaoui

3. [Catalyst Acceleration for First-order Convex Optimization:
from Theory to Practice](https://arxiv.org/pdf/1712.05654.pdf) H. Lin, J. Mairal, Z. Harchaoui

#### References Bibtex 
```
1. @inproceedings{lin2015universal,
  title={A universal catalyst for first-order optimization},
  author={Lin, Hongzhou and Mairal, Julien and Harchaoui, Zaid},
  booktitle={Advances in Neural Information Processing Systems},
  pages={3384--3392},
  year={2015}
}

2. @article{lin2016quickening,
  title={A Generic Quasi-Newton Algorithm for Faster Gradient-Based Optimization},
  author={Lin, Hongzhou and Mairal, Julien and Harchaoui, Zaid},
  journal={arXiv preprint arXiv:1610.00960},
  year={2016}
}

3. @article{2017arXiv171205654L,
   author = {{Lin}, H. and {Mairal}, J. and {Harchaoui}, Z.},
   title = "{Catalyst Acceleration for First-order Convex Optimization: from Theory to Practice}",
   journal = {arXiv peprints  arXiv:1712.05654},
   year={2017}
}
```
