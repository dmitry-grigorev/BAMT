import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import experiments2.auxiliary.auxiliary as ex2aux

from experiments3.random_DAG_generation import *
import random


def simulation(n_features, n_samples=2000, random_state=42, n_trials=20, transitive_mode=False):
    names = ['feature_' + str(i + 1) for i in range(n_features)]
    rde_acc, wde_acc = list(), list()
    asce_acc, dsce_acc = list(), list()
    wrong_acc = list()

    seed = random_state
    i = 0
    while i < n_trials:
        edges, into_degree, out_degree, position = DAGs_generate(n=n_features, max_out=2, random_state=seed)
        seed += 1
        if len(edges) == 0:
            continue
        bn_graph = nx.DiGraph()
        bn_graph.add_edges_from(edges)
        bn_graph.add_nodes_from(list(range(1, n_features + 1)))
        if bn_graph.number_of_nodes() != n_features:  # иногда получается больше положенного
            continue

        i += 1
        # plt.figure()
        # nx.draw_networkx(bn_graph, arrows=True, pos=position)
        # plt.show()

        adj_m = set_signs(bn_graph, seed) if not transitive_mode else get_adjacency_matrix_transitive(
            set_signs(bn_graph, seed), bn_graph)

        true_edges, signs = instantiate_gradation_relation(names, adj_m)

        data = build_dataset(adj_m, names, n_samples, seed)

        kresult = ex2aux.construct_by_kmeans(data, [])
        kbn = kresult['bn']
        kencoder, data_kdiscretized_enc = kresult['encoder'], kresult['disc_data']

        true_edges_ascend, true_edges_descend = [true_edges[i] for i in range(len(true_edges)) if signs[i] > 0], \
            [true_edges[i] for i in range(len(true_edges)) if signs[i] < 0]

        rde, wde = ex2aux.calculate_ratio(kbn.edges, true_edges), ex2aux.calculate_reversed_ratio(kbn.edges, true_edges)

        #print(true_edges_ascend)
        #print(true_edges_descend)

        # ratio of true edges corresponding to ascending relations in BN among all of those
        if len(true_edges_ascend) > 0:
            asce = ex2aux.calculate_ratio(kbn.edges, true_edges_ascend) + ex2aux.calculate_reversed_ratio(kbn.edges,
                                                                                                          true_edges_ascend)
            asce_acc.append(asce)
        # ratio of true edges corresponding to descending relations in BN among all of those
        if len(true_edges_descend) > 0:
            dsce = ex2aux.calculate_ratio(kbn.edges, true_edges_descend) + ex2aux.calculate_reversed_ratio(kbn.edges,
                                                                                                           true_edges_descend)
            dsce_acc.append(dsce)

        wronge = 1 - ex2aux.calculate_ratio(true_edges, kbn.edges) - ex2aux.calculate_reversed_ratio(true_edges,
                                                                                                     kbn.edges)

        rde_acc.append(rde)
        wde_acc.append(wde)
        wrong_acc.append(wronge)

        del kbn
        del edges
        del bn_graph
        del adj_m
        del data

        print(f"Stage {i + 1}", end='\r')

    answer = [sum(rde_acc) / n_trials, min(rde_acc), max(rde_acc),
              sum(wde_acc) / n_trials, min(wde_acc), max(wde_acc),
              sum(asce_acc) / len(asce_acc), min(asce_acc), max(asce_acc),
              sum(dsce_acc) / len(dsce_acc), min(dsce_acc), max(dsce_acc),
              sum(wrong_acc) / n_trials, min(wrong_acc), max(wrong_acc)
              ]

    print("mean, min and max precision on right direction: {:.3f} {:.3f} {:.3f} \n".format(*answer[:3]))
    print("mean, min and max precision on wrong direction: {:.3f} {:.3f} {:.3f} \n".format(*answer[3:6]))
    print("mean, min and max precision on ascending relations: {:.3f} {:.3f} {:.3f} \n".format(*answer[6:9]))
    print("mean, min and max precision on descending relations: {:.3f} {:.3f} {:.3f} \n".format(*answer[9:12]))
    print("mean, min and max precision on wrong edges among all in BN: {:.3f} {:.3f} {:.3f} \n".format(*answer[12:15]))
    return answer


def set_signs(G, random_state=42):
    np.random.seed(random_state)
    adj_m = nx.adjacency_matrix(G, nodelist=list(range(1, G.number_of_nodes() + 1))).todense().astype("float16")
    irows, icols = np.nonzero(adj_m)
    for i, j in zip(irows, icols):
        adj_m[i, j] = np.random.uniform(-2, 2)
    return adj_m


def get_adjacency_matrix_transitive(adj_m, bn_graph):
    # создаём матрицу линейных соотношений в транзитивном замыкании
    n = bn_graph.number_of_nodes()
    M = np.zeros_like(adj_m)
    for J in range(n, 0, -1):
        # посещаем каждую вершину в обратном обходе
        for j, i in nx.dfs_edges(bn_graph.reverse(), source=J):
            # от каждой обходим граф в глубину для связывания линейных соотношений
            if j == J:
                M[i - 1, J - 1] = adj_m[i - 1, j - 1]
            for j1, i1 in nx.bfs_edges(bn_graph.reverse(), source=i, depth_limit=1):
                # перемножаем
                M[i1 - 1, J - 1] += M[i - 1, J - 1] * adj_m[i1 - 1, j1 - 1]
    return M


def instantiate_gradation_relation(names, adj_m):
    true_edges = list()
    states = [0, 1, 2]
    signs = []
    nnzi, nnzj = np.nonzero(adj_m)
    for i, j in zip(nnzi, nnzj):
        sign = int(np.sign(adj_m[i, j]))
        true_edges += ([[names[i] + str(k), names[j] + str(m)] for k, m in zip(states, states[::sign])])
        signs += [sign]*3
    return true_edges, signs


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
