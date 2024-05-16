"""
use own evaluation codes - for new thresRO result
"""
import os
import glob

import numpy as np
import pandas as pd

import utils.evaluation as eval_func


SAVE_FOLDER = './res/structure_line_exp/NoOver_Md/'

GT_FOLDER = './Data/las-singleroof-GT/TXT_Zup_modify/'
GT_END = '_v'
PRED_FOLDER = './res/structure_line_exp/NoOver_Md/TRD_dataset_50_savemore/'
PRED_END = '_finCor_Geo'

LIST_ROOF_withGT = './config/sampled_roof_50.txt'

isSave_2_csv = True

save_name = os.path.join(SAVE_FOLDER, 'eval_res_'+os.path.basename(os.path.dirname(PRED_FOLDER))+'_mGT.csv')

check_list = pd.read_csv(LIST_ROOF_withGT, header=None, dtype='str')
check_list = sum(check_list.values.tolist(), [])


columns_name = ['roof_id',
                'precision', 'recall', 'vd_x', 'vd_y', 'vd_z',
                'vs_tp_num', 'vs_gt_num', 'vs_pred_num',
                'vs_tp']
all_res = pd.DataFrame([], columns=columns_name)
for fi,roof_id in enumerate(check_list):
    # read data
    gt_file   = os.path.join(GT_FOLDER, str(roof_id)+GT_END+'.txt')
    pred_file = os.path.join(PRED_FOLDER, str(roof_id) + PRED_END + '.txt')

    vs_pred = np.loadtxt(pred_file, delimiter=',')  # shape: (m, 3)
    vs_gt = np.loadtxt(gt_file, delimiter=',')  # shape: (n, 3)

    """Step 1: find corresponding pred_v for each gt_v"""
    v_gt_pred_dist = eval_func.find_v_gt_pred_dist(vs_gt, vs_pred) # [x_gt,y_gt,z_gt, x_pred,y_pred,z_pred, dist]
    # print(v_gt_pred_dist)

    """Step 2: get tp and num(tp), num(tp+fp), num(tp+fn)"""
    vs_tp, vs_tp_num, vs_gt_num, vs_pred_num = eval_func.stats_tp_gt_pred(v_gt_pred_dist, vs_gt, vs_pred, thrs_true=1)


    """Step 2: P and R for each roof"""
    precision, recall = eval_func.calc_p_r(vs_tp_num, vs_gt_num, vs_pred_num)

    """Step 3: VD(x), VD(y), VD(z) for each roof"""
    vd_x, vd_y, vd_z = eval_func.calc_vd_xyz(vs_tp)

    """Step 4: add to all"""
    # add each roof's eval res
    res_i = [roof_id, precision, recall, vd_x, vd_y, vd_z, vs_tp_num, vs_gt_num, vs_pred_num, vs_tp]
    res_i = pd.DataFrame([res_i], columns=columns_name)
    all_res = pd.concat([all_res, res_i], ignore_index=True)

"""Step fin.: calculate the overall evaluation result"""
vs_tp_num_all   = np.sum(all_res['vs_tp_num'])
vs_gt_num_all   = np.sum(all_res['vs_gt_num'])
vs_pred_num_all = np.sum(all_res['vs_pred_num'])

precision_overall, recall_overall = eval_func.calc_p_r(vs_tp_num_all, vs_gt_num_all, vs_pred_num_all)

vs_tp_all = np.vstack(all_res['vs_tp'].tolist())
vd_x_overall, vd_y_overall, vd_z_overall = eval_func.calc_vd_xyz(vs_tp_all)

"""Add overall res to all_res"""
res_overall = ['all', precision_overall, recall_overall, vd_x_overall, vd_y_overall, vd_z_overall,
               vs_tp_num_all, vs_gt_num_all, vs_pred_num_all, []]
res_overall = pd.DataFrame([res_overall], columns=columns_name)
all_res = pd.concat([all_res, res_overall], ignore_index=True)

"""Save all_res"""
if isSave_2_csv:
    all_res.to_csv(save_name, index=False)

print("precision:\t",precision_overall)
print("recall:\t",recall_overall)
print("vdxyz:\t",[vd_x_overall, vd_y_overall, vd_z_overall])




