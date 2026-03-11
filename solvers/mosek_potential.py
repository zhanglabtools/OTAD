"""LP solver for convex potential estimation using MOSEK."""

import numpy as np
from mosek.fusion import *

def LP(l, L, OT_input, OT_output):
    with Model("OTG") as M:
        n = OT_input.shape[1]
        d = OT_input.shape[0]
        
        c1 = 1/(2*(1-l/L))
        c2 = 1/L
        c3 = l
        c4 = 2*l/L
        C1 = c1*c2
        C2 = c1*c3
        C3 = c1*c4
        
        X = OT_input.transpose()
        T = np.zeros((n, d))
        Z = T + OT_output.transpose()
        
        U = M.variable("U", d, Domain.greaterThan(0))
        gamma = M.variable("gamma", 1, Domain.greaterThan(0))
        
        for s in range(d):
            for t in range(1):
                i = s*1 + t
                Xt = np.tile(np.reshape(X[:, i], (n, 1)), (1, d)) - X
                lt0 = np.diag(Z.transpose().dot(Xt))
                Zt = np.tile(np.reshape(Z[:, i], (n, 1)), (1, d)) - Z
                lt1 = lt0 + C1*np.sum(Zt*Zt, axis=0)
                lt2 = lt1 + C2*np.sum(Xt*Xt, axis=0)
                lt3 = lt2 - C3*np.diag(Zt.transpose().dot(Xt))
                lt3 = np.reshape(lt3, (d, 1))
                np.float64(lt3)
                
                At = -np.eye(d)
                At[:, i] = At[:, i] + 1
                
                if t == 0:
                    lt = lt3
                    A = At
                else:
                    A = np.vstack((A, At))
                    lt = np.vstack((lt, lt3))
                    print(t, lt.shape, A.shape)
            
            M.constraint(Expr.sub(Expr.mul(A, U), lt), Domain.greaterThan(0.0))
        
        M.constraint(Expr.sub(Var.repeat(gamma, d), U), Domain.greaterThan(0.0))     
        M.objective("obj", ObjectiveSense.Minimize, gamma)
        M.solve()
        sol = U.level()
    
    return sol
