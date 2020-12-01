# ML-and-LA-from-scratch
Some machine learning (ML) methods plus related numerical linear algebra (LA) and also quadratic programming algorithms to solve the ML optimization problems. This means that no implementation from external libraries is used except for numpy arrays and basic numpy operations on arrays, such as algebraic operations, matrix multiplication, etc. The algorithms are described in Theoretical-background.pdf.

Note that, neither the implementations are optimized nor are the chosen (especially LA) algorithms to solve the ML problems optimal.  
The considered LA algorithms are: QR-decomposition based on Gram-Schmidt proces and Housholder reflections, QR algorithm to determine eigenvalues (and vectors for symmetric matrices), singular-value decompostion, Cholesky decomposition, and forward and backward substitution. 
Quadratic programming is performed via the primal-dual interior-point method.  

The included machine-learning methods are: linear (least-squares and ridge) regression, non-negative least squares regression, orthogonal matching pursuit, kernel ridge regression, support vector machines, logistic regression, and principal component analysis.
