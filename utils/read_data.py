import os

import numpy as np
import pandas as pd

def read_pts(pts_file):
    with open(pts_file, 'r') as f:
        lines = f.readlines()
        pts = np.array([f.strip().split(' ') for f in lines], dtype=np.float64)
    return pts


def load_obj(obj_file):
    vs, edges = [], set()
    with open(obj_file, 'r') as f:
        lines = f.readlines()
    for f in lines:
        vals = f.strip().split(' ')
        if vals[0] == 'v':
            vs.append(vals[1:])
        else:
            obj_data = np.array(vals[1:], dtype=np.int).reshape(-1, 1) - 1
            idx = np.arange(len(obj_data)) - 1
            cur_edge = np.concatenate([obj_data, obj_data[idx]], -1)
            [edges.add(tuple(sorted(e))) for e in cur_edge]
    vs = np.array(vs, dtype=np.float64)
    edges = np.array(list(edges))
    return vs, edges


def load_obj_v2(obj_file):
    """
    load obj file with '#' annotation, vn information and face information in 'f'
    :param obj_file: file path of obj file
    :return:
    """
    vs, edges, face = [], set(), set()
    with open(obj_file, 'r') as f:
        lines = f.readlines()
    for f in lines:
        vals = f.strip().split(' ')
        # print('vals: ', vals)
        if vals[0] in ['#', 'vn']:
            continue
        elif vals[0] == 'v':
            # v in obj files: x, z, -y
            # original:
            # vs.append(vals[1:])
            #
            # new version:
            vs_i = [vals[1], vals[3], vals[2]]  # x, z, -y -> x, -y, z
            vs.append(vs_i)
        else:
            face_data = np.array(vals[1:])
            face_data = np.array(list(np.char.split(face_data, sep='//')),
                                 dtype='int') - 1  # [[vertices_No., face_No.]] # shape: (n, 2)
            edge_v = face_data[:, 0].reshape(-1, 1)  # shape: (n, 1)
            # face_No = face_data[:,1].reshape(-1, 1) # shape: (n, 1)
            idx = np.arange(len(edge_v)) - 1
            cur_edge = np.concatenate([edge_v, edge_v[idx]], -1)  # shape: (n,2) [[start_v, end_v], ...]
            [edges.add(tuple(sorted(e))) for e in cur_edge]
            # print('cur_edge: ', cur_edge)

    vs = np.array(vs, dtype=np.float64)
    vs[:, 1] = vs[:, 1] * (-1)  # -y -> y
    edges = np.array(list(edges))

    return vs, edges


def load_obj_Zup(obj_file):
    """
    load obj file with '#' annotation, vn information and face information in 'f'
    :param obj_file: file path of obj file
    :return:
    """
    vs, edges, face = [], set(), set()
    with open(obj_file, 'r') as f:
        lines = f.readlines()
    for f in lines:
        vals = f.strip().split(' ')
        # print('vals: ', vals)
        if vals[0] in ['#', 'vn']:
            continue
        elif vals[0] == 'v':
            # v in obj files: x, z, -y
            # original:
            # vs.append(vals[1:])
            #
            # new version:
            vs_i = [vals[1], vals[2], vals[3]]  # x, z, -y -> x, -y, z
            vs.append(vs_i)
        else:
            face_data = np.array(vals[1:])
            face_data = np.array(list(np.char.split(face_data, sep='//')),
                                 dtype='int') - 1  # [[vertices_No., face_No.]] # shape: (n, 2)
            edge_v = face_data[:, 0].reshape(-1, 1)  # shape: (n, 1)
            # face_No = face_data[:,1].reshape(-1, 1) # shape: (n, 1)
            idx = np.arange(len(edge_v)) - 1
            cur_edge = np.concatenate([edge_v, edge_v[idx]], -1)  # shape: (n,2) [[start_v, end_v], ...]
            [edges.add(tuple(sorted(e))) for e in cur_edge]
            # print('cur_edge: ', cur_edge)

    vs = np.array(vs, dtype=np.float64)
    # vs[:, 1] = vs[:, 1] * (-1)  # -y -> y
    edges = np.array(list(edges))

    return vs, edges


def obj_yup_2_vup(obj_file):
    with open(obj_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for f in lines:
        vals = f.strip().split(' ')
        if vals[0] in ['#', 'vn', 'f']:
            new_lines.append(f)
        elif vals[0] == 'v':
            # vs_i = [vals[1], vals[3], vals[2]]  # x, z, -y -> x, -y, z
            vx, vy, vz = vals[1], vals[3][1:], vals[2]  # x, z, -y -> x, y, z
            vs_line = f"v {vx} {vy} {vz}\n"
            new_lines.append(vs_line)
        else:
            raise Exception(f"uncontrolled type of obj line. the first keyword is {vals[0]}, "
                            f"which is not in ['#', 'v', 'vn', 'f']")

    return new_lines


def save_obj(out_file, lines, mode="w+"):
    with open(out_file, mode) as out_objfile:
        out_objfile.writelines(lines)


def save_obj_methodout(out_file, vs, fs):
    # save vs and vn info. at first
    np.savetxt(out_file, vs, delimiter=' ', fmt='%s')
    # then save face info. after vs and vn info.
    with open(out_file, 'a') as out_file:
        out_file.writelines(fs)


def save_obj_methodout_unorgdata(out_file, fin_geo_corners, faces):
    v_head = np.full(shape=(fin_geo_corners.shape[0], 1), fill_value='v', dtype='str')
    fin_geo_corners_str = np.char.mod("%.6f", fin_geo_corners)
    vs_geo = np.hstack([v_head, fin_geo_corners_str])

    # organize faces information
    fs = []
    for i in range(len(faces)):
        f_str = "f " + " ".join(map(str, np.array(faces[i]) + 1)) + "\n"
        fs.append(f_str)

    save_obj_methodout(out_file, vs=vs_geo, fs=fs)

def save_dataframe(out_file, data: pd.DataFrame):
    data.to_csv(out_file, index=False)


# NOTE: added part for load gt txt file for vertice and edge eval
def load_txt_rfst_Zup(txt_folder:str, roof_id) -> (np.ndarray, np.ndarray):
    vs_file_path = os.path.join(txt_folder, f"{roof_id}_v.txt")
    es_file_path = os.path.join(txt_folder, f"{roof_id}_e.txt")

    vs = np.loadtxt(vs_file_path, delimiter=',')  # shape: (n, 3)
    es = np.loadtxt(es_file_path, delimiter=',').astype("int")  # shape: (n, 3)

    return vs, es

def load_obj_rfst_Zup(obj_file:str) -> (np.ndarray, np.ndarray):
    """
    load obj file with '#' annotation, vn information and face information in 'f'
    :param obj_file: file path of obj file
    :return:
    """
    vs, edges, face = [], set(), set()
    with open(obj_file, 'r') as f:
        lines = f.readlines()
    for f in lines:
        vals = f.strip().split(' ')
        # print('vals: ', vals)
        if vals[0] in ['#', 'vn']:
            continue
        elif vals[0] == 'v':
            # v in obj files: x, y, z
            # original:
            # vs.append(vals[1:])
            #
            # new version:
            vs_i = [vals[1], vals[2], vals[3]]
            vs.append(vs_i)
        else:
            face_data = np.array(vals[1:])
            face_data = np.array(list(np.char.split(face_data, sep='//')),
                                 dtype='int') - 1  # [[vertices_No., face_No.]] # shape: (n, 2)
            edge_v = face_data[:, 0].reshape(-1, 1)  # shape: (n, 1)
            # face_No = face_data[:,1].reshape(-1, 1) # shape: (n, 1)
            idx = np.arange(len(edge_v)) - 1
            cur_edge = np.concatenate([edge_v, edge_v[idx]], -1)  # shape: (n,2) [[start_v, end_v], ...]
            [edges.add(tuple(sorted(e))) for e in cur_edge]
            # print('cur_edge: ', cur_edge)

    vs = np.array(vs, dtype=np.float64)
    # vs[:, 1] = vs[:, 1] * (-1)  # -y -> y
    edges = np.array(list(edges))

    return vs, edges




