import numpy as np
from linear_algebra import svd

def qp(Q, p, G, h, A_in, b_in, mu=.00001, alpha=0.1, max_steps_qp=80, **kwargs):
    """Needs to be checked, optimized, rerwitten... and is probably even erroneous.
       Quadratic programming. Solving the problem:
         min_x      x.T Q x + p.T x
       subject to   A_in x = b_in  and  G x <= h  x => 0.
       
       The implementation which adjusts parameters for fast convergence, 
       e.g. mu and alpha, is missing. Therefore, convergence needs to be manually
       checked and paramters retuned for every new problem. 

    Parrameters
    -----------
    Q, p, G, h, A_in, b_in: arrays

    mu : float 
        Lagrange multiplier in logarithmic barrier function for non-negative variables,
        i.e. x but also slack variables that allow to transform the in equality in 
        G x <= h to the form of A_in x = b_in.

    alpha : float
        learning rate

    max_steps_qp : int
        maximum number of iterations.

    kwargs: kwargs of svd for inverting Jacobian.

    Return
    ------
    x: array
        Solution to the optimization problem
            
    """

    n = Q.shape[0]
    g = G.shape[0]
    m = A_in.shape[0] + G.shape[0]
    
    
    A = np.block([[A_in, np.zeros((len(A_in), g))], [G, np.eye(g)]])
    b = np.append(b_in, h)
    
    # init vars
    xc = np.ones(n+g)
    lam = np.ones(A.shape[0])
    s = np.ones(n+g)

    for i in range(max_steps_qp):
        Qxp = np.dot(Q, xc[: n]) + p.flatten()
        Qxp = np.pad(Qxp, (0,g), 'constant')
        F1 = Qxp - np.dot(A.T, lam) - s
        F2 = np.dot(A, xc) - b
        F3 = xc * s - mu
        F = - np.concatenate([F1, F2, F3])

        Qc = np.zeros((n+g, n+g))
        Qc[:n, :n] = Q.copy()
        
        # Jacobian
        J = np.block([[Qc, -A[:, :].T,        -np.eye(n+g)],
                      [A[:, :], np.zeros((m, m)), np.zeros((m, n+g))],
                      [np.diag(s), np.zeros((n+g, m)), np.diag(xc[:]) ]])
        
        # solve linear system J d = F, 
        # this is the most expensive part 
        # and actually needs a good solver,
        # e.g. d = np.linalg.inv(J) @ F
        U, S, V = svd(J, **kwargs)
        d = V / S @ U.T @ F
  #      d = np.linalg.inv(J) @ F

        d2 = alpha* d
        xc[:n+g] += d2[:n+g]
        s += d2[-n-g:]
        lam += d2[n+g:-n-g]

        print("### quadratic programming iteration %s/%s" %(i+1, max_steps_qp))
        #print(xc[:n])
    print("WARNING: There is no implementation in the quadratic programming (qp) code for checking")
    print("         the convergence criteria. Parameters need always to be tuned and checked manually.")
    print("         Furthremore, for increasing the speed of the code, replace the part in qp")
    print("         where the linear system J d = F is solved by d = np.linalg.inv(J) @ F.")
    #print(xc)
    return  xc[:n]

if __name__ == '__main__':

    Q = 2*np.array([ [2, .5], [.5, 1] ])
    p = np.array([1.0, 1.0])
    G = np.array([[-1.0,0.0],[0.0,-1.0]])
    h = np.array([0.0,0.0])
    A = np.array([[1.0, 1.0]])
    b = 1.0


    print("minimize     2 x1^2 + x2^2 + x1 x2 + x1 + x2")
    print("subject to   x1 => 0")
    print("             x2 => 0")
    print("             x1 + x2 = 1")
    print("solution:")
    x = qp(Q, p, G, h, A, b, max_steps_qp=100)
    print(x)
