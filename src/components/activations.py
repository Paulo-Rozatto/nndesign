from math import exp

def hardlim(n): # hard limit
    if(n < 0): 
        return 0
    return 1

def hardlims(n): # symetrical hard limit 
    if(n < 0): 
        return -1
    return 1

def lin(n): # linear
    return n

def satlin(n): # saturating linear
    if(n < 0):
        return 0
    return min(n, 1)

def satlins(n):
    if(n < -1):
        return -1
    return min(n, 1)

def logsig(n): # Log-Sigmoid
    return  1 / (1 + exp(-n))

def tansig(n): # Hyperbolic Tangent Sigmoid
    return (exp(n) - exp(-n)) / (exp(n) + exp(-n))

def poslin(n): # Positive Linear or relu
    if(n < 0):
        return 0
    return n

# TODO: competitive
# a = 1neuron with max n
# a = 0all other neurons

__dict__={
    "hardlim": hardlim,
    "hardlims": hardlims,
    "linear": lin,
    "satlin": satlin,
    "satlins": satlins,
    "logsig": logsig,
    "tansig": tansig,
    "poslin": poslin
}