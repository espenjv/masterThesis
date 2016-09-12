import numpy as np



def gaussQuadrature(p1,p2,p3,p4,fn,Nq):
    # Computes the numerical integration of function fn over the 1x1 square
    # with points p1,p2,p3,p4
    # Parameters: p1: upper left corner
    #             p2: upper right corner
    #             p3: lower left
    #             p4: lower right
    #             fn: function to be integrated
    #             Nq: number of integration points
    # Returns     I: integral value

    assert p1[0]==p3[0], ""

    A = p1[0] + p2[0]




    return I
