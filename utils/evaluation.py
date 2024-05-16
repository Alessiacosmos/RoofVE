import numpy as np

def norm_xyz(xyz, x_min, y_min, z_min):
    x_norm = xyz[:,0]  - x_min
    y_norm = xyz[:, 1] - y_min
    z_norm = xyz[:, 2] - z_min

    xyz_norm = np.array([x_norm, y_norm, z_norm]).T
    return xyz_norm

def find_closest_v_pred(row, vs_pred):
    """
    calculate the euclid distance between each gt_vertex and all pred_vertices
    and then, find (the closest pred_vertex)-'v_clst_pred' of this (gt_vertext)-'v_gt'
    return 'v_clst_pred' (the closest pred_vertex)
    :param row:     gt_vertex,     shape: (1, 3)
    :param vs_pred: pred_vertices, shape: (m, 3)
    :return:
    """
    # vs_dists = np.sqrt(np.sum(np.power(vs_pred-row,2),axis=1))
    vs_dists = np.sqrt(np.power(vs_pred[:,0]-row[0], 2) +
                       np.power(vs_pred[:,1]-row[1], 2) +
                       np.power(vs_pred[:,2]-row[2], 2))   # shape: (m,)
    # find min_dist's pred vertex
    idx_min_dist = np.where(vs_dists==np.min(vs_dists))[0][0]

    # get closest pred_vertex of this gt_vertex
    vs_pred_with_dists = np.concatenate([vs_pred, vs_dists.reshape(-1,1)], axis=1)
    # print(vs_pred_with_dists)
    v_clst_pred = vs_pred_with_dists[idx_min_dist, :]

    return v_clst_pred


def find_v_gt_pred_dist(vs_gt, vs_pred):
    """
    find each gt_v's corresponding pred_v， and concat [gt, pred, their dist] as the result to return
    :param vs_gt:
    :param vs_pred:
    :return:
    """
    v_corr_pred_with_dist = np.apply_along_axis(func1d=find_closest_v_pred, axis=1, arr=vs_gt, vs_pred=vs_pred)
    v_gt_pred_dist = np.concatenate([vs_gt, v_corr_pred_with_dist], axis=1)

    return v_gt_pred_dist

def stats_tp_gt_pred(v_gt_pred_dist, vs_gt, vs_pred, thrs_true=1):
    """
    statistic num(tp), num(tp+fp), num(tp+fn)
    :param v_gt_pred_dist:
    :param vs_gt:
    :param vs_pred:
    :param thrs_true:
    :return:
    """
    vs_tp = v_gt_pred_dist[v_gt_pred_dist[:, -1] <= thrs_true]
    vs_tp_num = vs_tp.shape[0]
    vs_gt_num = vs_gt.shape[0]
    vs_pred_num = vs_pred.shape[0]

    return vs_tp, vs_tp_num, vs_gt_num, vs_pred_num


def calc_p_r(vs_tp_num, vs_gt_num, vs_pred_num):
    """
    calculate precision and recall
    :param vs_tp_num:
    :param vs_gt_num:
    :param vs_pred_num:
    :param thrs_true:
    :return:
    """

    precision = vs_tp_num / vs_pred_num
    recall = vs_tp_num / vs_gt_num

    return precision, recall


def calc_vd_xyz(vs_tp):
    """
    calculate vd_x, vd_y, vd_z
    :param vs_tp:
    :return:
    """
    vd_x, vd_y, vd_z = np.mean(np.abs(vs_tp[:, :3] - vs_tp[:, 3:-1]), axis=0)
    return vd_x, vd_y, vd_z


