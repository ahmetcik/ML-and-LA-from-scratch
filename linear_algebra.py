import numpy as np

def qr_decomposition(A, modified_gs=True, tol_qr=1e-8):
    """QR decomposition based on Gram-Schmidt process. The used convention
       is Q being m X r matrix with m x n matrix A and r = rank(A).

    Parameters
    ----------
    A : array, [n, m]
        Matrix to be factored.

    modified_gs : bool, default True
        If True, the 'modified' Gram-Schmidt implementaion is used which 
        is numeracially more stable, however, computationally more demanding.
    
    tol_qr: float
        If every value of orthogonal projection of new vector to be orthogonlized
        on span of orthogonazed vectors is smaller tol, then the new vector is
        considered linearly dependent and is skipped.

    Return:
    -------
        Q, R: (array, array), ([m, r], [r, n])
            Q and R...
    """

    m, n = A.shape
    
    # get Q
    Q = np.empty((m, 0))
    for i in range(n):
        x = A[:, i]

        # Gram-Schmidt step
        if i == 0:
            x = A[:, i]
        elif modified_gs:
            for b in Q.T:
                x = x - np.dot(b, x)*b
        else:
            x = A[:, i] - Q.dot(Q.T).dot(A[:, i])

        if any(abs(v) > tol_qr for v in x):
            x = x / np.linalg.norm(x)
            Q = np.column_stack((Q, x))

    # get R
    R = np.zeros((Q.shape[1], n))
    for i in range(Q.shape[1]):
        R[i, i:] = np.dot(Q[:, i], A[:, i:])
    #or R = np.dot(Q.T, A)
    
    return Q, R



def qr_decomposition_housholder(A_in):
#def qr_decomposition(A_in):
    """QR decomposition based on Householder reflections.
       Q is a square matrix of length t = min(m-1, n) where A.shape = m, n.
       Implementation needs to be rechecked!!!

    Parameters
    ----------
    A : array, [m, n]
        Matrix to be factored.

    Return:
    -------
        Q, R: (array, array)
            Q and R...
    """
    A = A_in.copy()
    m, n = A.shape
    t = min(m-1, n)


    I = np.eye(A.shape[0], A.shape[0])
    QQ = I.copy()
    for i in range(t):
        x = A[:, 0]
        alpha = - np.sign(x[0]) *np.linalg.norm(x)
        u = x + alpha * I[0,:m-i]
        v = u / np.linalg.norm(u)
        #if np.any(np.isnan(v)):

        Q = I.copy()
        Q[i:,i:] -= 2 * np.dot(v[:, np.newaxis], v[np.newaxis])

        QQ = np.dot(Q, QQ)
        A = Q[i+1:,i+1:]

    R = np.dot(QQ, A_in)
    Q = QQ.T
    return Q, R

def eigen(A, tol_eigen_value=1e-30, max_steps=1000000, **kwargs):
    """Calculate eigenvalues and eigenvectors of symmetric matrix A
       using QR algorithm. For non-symmetric A, eigenvalues might still 
       be correct.

    Parameters
    ----------
    A : array, [n, n]
        Matrix to be considered.
    
    tol_eigen_value: float
        Convergence tolerance of all eigenvalues.

    max_steps: int
        Maximum number of iterations in the QR algorithm.a

    kwargs: kwargs of qr_decomposition

    Return:
    -------
        eigen_values, eigen_vectors
            eigen_vectors are given as columns of the returned matrix 
    """

    if not np.allclose(A, A.T):
        print("WARNING: matrix is not symmetric. Eigenvectors should not be used.")
    n = A.shape[0]
    B = A.copy()
    Q = np.eye(n)
    for i in range(max_steps):
        q, r = qr_decomposition(B, **kwargs)
        B = np.dot(r, q)
        Q = np.dot(Q, q)

        if i == 0:
            # number of eigenvalues depends on rank(A) or q.shape
            eigen_values_last = np.full(q.shape[1], np.inf)
        elif i == max_steps -1:
            print("WARNING: Eigenvalues not converged.")
        else: 
            eigen_values = np.diag(B)
            abs_diff = abs(eigen_values - eigen_values_last)
            if all(v < tol_eigen_value for v in abs_diff):
                break
            eigen_values_last = eigen_values
    eigen_vectors = Q
    return eigen_values, eigen_vectors

def svd(A, **kwargs):
    """Singular value decomposition for matrix A.
       
    Parameters
    ----------
    A : array, [n, m]
        Matrix to be considered.
    
    kwargs: kwargs of eigen function and qr_decomposition
    
    Return:
    -------
        U, S, V
        such that A = np.dot(U * S, V.T).
    """
    
    ATA = np.dot(A.T, A)
    
    S, V = eigen(ATA, **kwargs)
    S = np.sqrt(S)
    U = np.dot(A, V)/ S
    return U, S, V


def pseudo_inverse(A, **kwargs):
    """Calculate pseudo inverse using singular value decomposition."""

    U, S, V = svd(A, **kwargs)
    return (V/S).dot(U.T)


def cholesky_decomposition(A):
    """Calculate lower triangular matrix of Cholesky decomposition"""

    n = A.shape[0]
    L = np.zeros((n, n)) 
        
    indices = ((i, j) for j in range(n) for i in range(n) if i > j-1)
    for i, j in indices:
        if i == j:
            L[i, i] = np.sqrt(A[i,i] - np.linalg.norm(L[i, :j])**2)
        else:
            L[i, j] = 1. / L[j, j] * (A[i, j] - np.dot(L[i, :j], L[j, :j]))
    return L

def solve_triangular(A, b, lower=True):
    """Perform forward or backward substitution depending on
       lower=True or lower=False respectively."""

    n = A.shape[0]
    indices = range(n) if lower else reversed(range(n))
    
    x = np.zeros(n)
    for i in indices:
        if lower:
            dot = np.dot(A[i, :i], x[:i])
        else:
            dot = np.dot(A[i, i:], x[i:])
        x[i] = (b[i] - dot) / A[i, i]
    return x

