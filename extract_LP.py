from mip import *
import math

def create_LP(B: list[list[int]]) -> Dict:
    """
    Linear program that simplifies a frontier given by input adjacency matrix `B` with a global gate.
    """
    rows_B = len(B)
    cols_B = len(B[0])
    m = Model(solver_name=GRB)

    # Variable definitions
    x = [[m.add_var('x_{},{}'.format(i, j), var_type=BINARY) for j in range(cols_B)] for i in range(rows_B)] # Encodes simplified input matrix
    y = [m.add_var(var_type=BINARY) for i in range(rows_B)] # Helper variable to create the constraint z[i]=1 if row i is extractable, 0 otherwise
    z = [m.add_var(var_type=BINARY) for i in range(rows_B)] # Encodes no. of extractable rows
    k = [[m.add_var('k_{},{}'.format(i,j), var_type=BINARY) for j in range(rows_B)] for i in range(rows_B)] # helper variable to linearize variable multiplication
    t = [[m.add_var('t_{},{}'.format(i, j), var_type=INTEGER) for j in range(cols_B)] for i in range(rows_B)] # helper variable to linearize XORs
    G = [[m.add_var('g_{},{}'.format(i,j), var_type=BINARY) for j in range(rows_B)] for i in range(rows_B)] # Encodes global gates

    # Constraint: at least one extractable vertex
    m += xsum(z[i] for i in range(len(z))) >= 1 # Sum_i z_i >= 1
    
    # Constraint: z[i]=1 if row i is extractable, 0 otherwise
    for i in range(rows_B):
        m += xsum(x[i][j] for j in range(cols_B)) -1 >= y[i]
        m += xsum(x[i][j] for j in range(cols_B)) -1 <= cols_B * y[i]
        m += y[i] + z[i] == 1

    # Constraint: values of x_ij come from multiplying input B by global gates. We need t and extra constraints to encode XORs
    for i in range(rows_B):
        for j in range(cols_B):

            m += t[i][j] >= 0
            m += t[i][j] <= math.floor(rows_B/2) # constrain `t` just to make the search space smaller

            m += x[i][j] >= 0
            m += x[i][j] <= 1
            m += x[i][j] == xsum(g*b for g,b in zip(G[i][:],[_[j] for _ in B])) - 2*t[i][j]

    # Constraint: global gates are made of commuting CNOTs
    # Create variable k[i] that represents \Prod_{j\ne i} 1 - G[i][j]
    for i in range(len(G)):
        for j in range(len(G[i])):
            if i == j:
                m += G[i][j] == 1
            else:
                for _ in range(len(G[j])):
                    if _ != j:
                        m += k[i][j] <= 1 - G[j][_]
                m += k[i][j] >= xsum(1 - G[j][_] for _ in range(len(G[j])) if _ != j) - (len(G[j])-1 - 1)
                m += (1-G[i][j]) + k[i][j] >= 1

    # Count number of CNOTs, used as penalty
    n_cnots = xsum(G[i][j] for j in range(len(G[i])) for i in range(len(G)) if i!=j)

    # Objective: maximize extractable vertices and minimize the number of used CNOTs
    m.objective = maximize(rows_B * xsum(z[i] for i in range(len(z))) - n_cnots )

    return {'model': m, 'gms': G, 'X': x, 'z': z}