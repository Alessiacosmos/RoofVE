"""
@File           : roof_graph.py
@Time           : 1/10/2023 4:26 PM
@Author         : Gefei Kong
@Description    : as below
--------------------------------------------------------------
create graph for roof structure, to find the cycles in the roof structure, and finally create faces info.
modify the problem of find_fin_cycles()
"""


import numpy as np
import utils.useful_func as ufunc

def create_graph(all_edges: np.ndarray) -> dict:
    key_nodes = np.unique(all_edges)

    graph = {}
    for i, key in enumerate(key_nodes):
        value_idx = np.where(all_edges == key) # get the row idxes (edges) which include key
        value = np.unique(all_edges[value_idx[0],:]) # extract all keys in these edges
        value = value[value!=key] # remove the key itself
        graph[key] = value.tolist()

    return graph


def dfs(graph: dict, start:int, end:int):
    """
    code link: https://stackoverflow.com/questions/40833612/find-all-cycles-in-a-graph-implementation
    :param graph:
    :param start:
    :param end:
    :return:
    """
    fringe = [(start, [])]
    while fringe:
        state, path = fringe.pop()
        if path and state == end:
            yield path
            continue
        for next_state in graph[state]:
            if next_state in path:
                continue
            fringe.append((next_state, path+[next_state]))


def find_cycles(graph:dict) -> list:
    """
    code link: https://stackoverflow.com/questions/40833612/find-all-cycles-in-a-graph-implementation
    :param graph:
    :return:
    """
    cycles = [[node] + path for node in graph for path in dfs(graph, node, node)]


    return cycles


def cycle_clean(cycles):
    """
    # remove cycles with cycle_number <=3: such element is not a polygon.
    # at the same time, sort vertices and rm reduplicated cycles, even this making the structure is not a cycle: to check whether the cycle has been considered.
    :param cycles:
    :return:
    """
    cycles_rm3_unique = []
    cycles_rm3_unique_sort = []
    for ci in cycles:
        if len(ci) <= 3:
            continue

        ci = ci[:-1]
        ci_sorted = sorted(ci)
        if ci_sorted not in cycles_rm3_unique_sort:
            cycles_rm3_unique.append(ci)
            cycles_rm3_unique_sort.append(ci_sorted)

    return cycles_rm3_unique, cycles_rm3_unique_sort


def count_subcycle_num(cycles_rm3_uniq_sort):
    """
    count the number of a cycle appeared in other cycles.
    :return:
    """

    cnt_cycle = []
    subset_matrix = []
    for i in range(len(cycles_rm3_uniq_sort)):
        # if i>1:
        #     break
        cycle_i = cycles_rm3_uniq_sort[i]

        # other cycles
        other_cycles = cycles_rm3_uniq_sort[:i] + [[]] + cycles_rm3_uniq_sort[(i+1):] # [[]]: Placeholder, representing cycle_i itself
        # check whether cycle_i is a true subset of other_cycles
        is_subset = [set(cycle_i) < set(other_c) for other_c in other_cycles]

        cnt_cycle.append(sum(is_subset)) # if is_subset else 0
        subset_matrix.append(is_subset)

    return subset_matrix, cnt_cycle


def find_all_cycles(all_edges):
    all_edges = np.array(all_edges)
    graph = create_graph(np.array(all_edges))
    # print("edge graph:\n", graph)
    cycles = find_cycles(graph)

    # sort cycles
    cycles = sorted(cycles)
    cycles_rm3_uniq, cycles_rm3_uniq_sort = cycle_clean(cycles)
    subset_matrix, cnt_cycle = count_subcycle_num(cycles_rm3_uniq_sort)
    return cycles_rm3_uniq, subset_matrix, cnt_cycle


def find_max_cycles(all_cycles: list) -> list:
    # iteratively find all unconnected cycles ###
    edge_vertices = np.unique(sum(all_cycles, []))
    not_considered_vertices = edge_vertices
    # print(edge_vertices)
    max_cycles = []
    iteri, max_iter = 0, 100
    while len(all_cycles) != 0 and iteri < max_iter:
        # print(iteri, all_cycles)
        len_cycle = [len(_) for _ in all_cycles]
        cand_face_idx = len_cycle.index(max(len_cycle))
        cycle_maxlen = all_cycles[cand_face_idx]

        max_cycles.append(cycle_maxlen)

        considered_vertices_i = np.unique(cycle_maxlen)
        not_considered_vertices = np.setdiff1d(not_considered_vertices, considered_vertices_i)
        # print("not_considered_vertices: ",not_considered_vertices)

        all_cycles = [c for c in all_cycles if len(set(c).intersection(not_considered_vertices)) != 0]


        iteri += 1

    return max_cycles


def find_max_cycles_v2(v_coords: np.ndarray, all_cycles: list) -> list:
    """
    find the max_area cycle
    :param v_coords:
    :param all_cycles:
    :return:
    """

    # iteratively find all unconnected cycles ###
    # edge_vertices = np.unique(sum(all_cycles, []))
    # not_considered_vertices = edge_vertices
    # print(edge_vertices)
    max_cycles = []
    iteri, max_iter = 0, 100
    while len(all_cycles) != 0 and iteri < max_iter:
        # print(iteri, all_cycles)
        """update here"""
        # len_cycle = [len(_) for _ in all_cycles]
        # cand_face_idx = len_cycle.index(max(len_cycle))
        # cycle_maxlen = all_cycles[cand_face_idx]
        area_cycle = [ufunc.calc_area(_, v_coords) for _ in all_cycles]
        cand_face_idx = area_cycle.index(max(area_cycle))
        cycle_maxarea = all_cycles[cand_face_idx]

        max_cycles.append(cycle_maxarea)
        all_cycles.pop(cand_face_idx)
        inter_per_cycle = [ufunc.calc_intersect_vs(_, cycle_maxarea, v_coords) for _ in all_cycles]

        all_cycles = [all_cycles[i] for i in range(len(all_cycles)) if inter_per_cycle[i]<1] # only consider tris not 100% overlapped.
        iteri += 1

    return max_cycles

















