import numpy as np
import networkx as nx
import  matplotlib.pyplot as plt
import pandas as pd
import experiments2.auxiliary.auxiliary as ex2aux

from experiments3.random_DAG_generation import *
import random


def simulation(n_features, n_samples=2000, random_state=42, n_trials=20):
    names = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']
    rde_acc, wde_acc = list(), list()
    for i in range(n_trials):
        seed = random_state + i

        edges, into_degree, out_degree, position = DAGs_generate(n=n_features, max_out=5, random_state=seed)
        bn_graph = nx.DiGraph()
        bn_graph.add_edges_from(edges)
        bn_graph.add_nodes_from(list(range(1, n_features+1)))

        plt.figure()
        nx.draw_networkx(bn_graph, arrows=True, pos=position)
        plt.show()

        adj_m = set_signs(bn_graph, seed)

        true_edges = instantiate_gradation_relation(names, adj_m)

        data = build_dataset(adj_m, names, n_samples, seed)

        kresult = ex2aux.construct_by_kmeans(data, [])
        kbn = kresult['bn']
        kencoder, data_kdiscretized_enc = kresult['encoder'], kresult['disc_data']

        rde, wde = ex2aux.calculate_ratio(kbn.edges, true_edges), ex2aux.calculate_reversed_ratio(kbn.edges, true_edges)

        rde_acc.append(rde)
        wde_acc.append(wde)

        del kbn
        del edges
        del bn_graph
        del adj_m

        print(f"Stage {i+1}", end='\r')
    print("mean, min and max precision on right direction: {:.3f} {:.3f} {:.3f} \n".format(sum(rde_acc)/n_trials, min(rde_acc), max(rde_acc)))
    print("mean, min and max precision on wrong direction: {:.3f} {:.3f} {:.3f} \n".format(sum(wde_acc) / n_trials,
                                                                                           min(wde_acc), max(wde_acc)))


def set_signs(G, random_state=42):
    np.random.seed(random_state)
    adj_m = nx.adjacency_matrix(G, nodelist=list(range(1, G.number_of_nodes() + 1))).todense().astype("float16")
    irows, icols = np.nonzero(adj_m)
    for i, j in zip(irows, icols):
        adj_m[i, j] = np.random.uniform(-2, 2)
    return adj_m


def instantiate_gradation_relation(names, adj_m):
    true_edges = list()
    states = [0, 1, 2]
    nnzi, nnzj = np.nonzero(adj_m)
    for i, j in zip(nnzi, nnzj):
        sign = int(np.sign(adj_m[i, j]))
        true_edges += ([[names[i] + str(k), names[j] + str(m)] for k, m in zip(states, states[::sign])])
    return true_edges


def build_dataset(adj_m, names, n_samples, random_state=42):
    m = adj_m.shape[0]
    np.random.seed(random_state)
    data = pd.DataFrame(columns=names[:m], data=np.zeros(shape=(n_samples, m)))
    for i in range(m):
        nnz = np.nonzero(adj_m[:, i])[0]
        if nnz.shape[0] == 0:
            data[names[i]] = np.random.normal(0, 3, size=n_samples)
        else:
            for j in nnz:
                data[names[i]] += adj_m[j, i] * data[names[j]] + np.random.normal(0, 0.7, size=n_samples)
    return data
