"""QCQP solver for OT map computation using MOSEK."""

import numpy as np
from mosek.fusion import *

def QCQP(l, L, OT_input, OT_output, BP, x):
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
        
        U = BP
        x = x + np.zeros((1, n))
        x = np.reshape(x, (n, 1))
        x = np.tile(x, (1, d))
        X1 = x - X
        X2 = X1.transpose()
        gamma2 = np.sum(X1*X1, axis=0)
        
        A = (1 + C3)*np.diag(Z.transpose().dot(X1)) + C2*gamma2
        
        v = M.variable("v", 1, Domain.unbounded())
        g = M.variable("g", [n, 1], Domain.unbounded())
        gamma1 = M.variable("gamma1", d, Domain.unbounded())
        
        M.constraint(Expr.add(Expr.sub(Expr.sub(Expr.sub(Var.vrepeat(v, d), U),
                              A), Expr.mul(C1, gamma1)),
                    Expr.mulDiag(X2, Var.repeat(g, 1, d))), Domain.greaterThan(0.0))
        
        M.constraint(Expr.hstack(Expr.mul(0.5, Expr.ones(d)), gamma1, Expr.transpose(Expr.sub(Var.repeat(g, 1, d), Z))), Domain.inRotatedQCone())
        
        M.objective("obj", ObjectiveSense.Minimize, v)
        M.solve()
        solv = v.level()
        solg = g.level()
    
    return solv, solg
