import numpy as np

def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)      
    n_edge = H.shape[1]  

    # the weight of the hyperedge
    W = np.ones(n_edge)   

    # the degree of the node
    DV = np.sum(H * W, axis=1)  
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)     

    invDE = np.mat(np.diag(np.power(DE, -1)))    
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))  
    W = np.mat(np.diag(W))      
    H = np.mat(H)               
    HT = H.T

    if variable_weight:         
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2    
        hyper_egde=invDE * HT * DV2
        return G,hyper_egde