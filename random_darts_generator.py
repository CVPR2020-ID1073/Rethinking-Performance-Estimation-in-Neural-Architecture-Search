import sys
from models.darts_cell import *
import genotypes as gt
import numpy as np

PRIMITIVES = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 
              'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'none']

def generate_random_structure(node = 4, k =2):
    total_edge = sum(list(range(2, node + 2))) * 2
    cell_edge = int(total_edge / 2)
    num_ops = len(PRIMITIVES)
    weight = np.random.randn(total_edge, num_ops)
    theta_norm = utils.darts_weight_unpack(weight[0:cell_edge], node)
    theta_reduce = utils.darts_weight_unpack(weight[cell_edge:], node)
    gene_normal = gt.parse_numpy(theta_norm, k=k)
    gene_reduce = gt.parse_numpy(theta_reduce, k=k)
    concat = range(2, 2+node)
    return gt.Genotype(normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat)


if __name__ == '__main__':
    file_name = 'random_darts_architecture.txt'
    file = open(file_name, 'w+')
    for i in range(100):
        graph = generate_random_structure()
        file.write(str(graph)+'\n')
    file.close()