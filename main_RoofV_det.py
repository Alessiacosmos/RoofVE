"""
@File           : main_RoofV_det.py
@Time           : 5/16/2023 2:53 PM
@Author         : Gefei Kong
@Description    : as below
--------------------------------------------------------------

"""


"""
test the result of removing the consideration of overlap in a cluster(原Rule6)
"""
import os
import glob

import laspy
import numpy as np
import pandas as pd

import utils.module_voxelizaton as mdl_voxel
import utils.visualization as visual
import utils.module_findStruct_v3_NoOver as mdl_fdStruct_v3_noOver

import utils.read_data as read_data

OPEN_FOLDER = './Data/las-singleroof-50/'
PCD_TYPE    = '.las'
StructP_FOLDER =  'structure_line_exp/NoOver_Md/TRD_dataset_savemore/'# ''
NEED_CHECKs = './config/sampled_roof_50.txt'

GRID_SIZE   = 0.5
max_search_radius_param = +1 # -1 means: (max(z)-1)/2

SAVE_FOLDER = f'structure_line_exp/NoOver_Md/TRD_dataset_50_savemore/'

pcd_list = glob.glob(os.path.join(OPEN_FOLDER, '*'+PCD_TYPE))

if not os.path.exists('./res/'+SAVE_FOLDER):
    os.makedirs('./res/'+SAVE_FOLDER)

if NEED_CHECKs != '': # if there are some specific roofs pcds need to be paid more attention
    check_list = pd.read_csv(NEED_CHECKs, header=None, dtype='str')
    check_list = sum(check_list.values.tolist(), [])
else:
    check_list = [] # pd.DataFrame([])

print(check_list)
print(f"{len(check_list)} roofs will be checked.")

""" Read data =================================================================="""

for fi,file_path in enumerate(pcd_list):
    if os.path.basename(file_path).replace('.las', '') not in check_list:
        continue

    print(f'=====================find structure for {fi}: {os.path.basename(file_path)}=====================')
    pcd_las = laspy.read(file_path)
    xyz = np.vstack((pcd_las.x, pcd_las.y, pcd_las.z)).transpose()  # n*3, n is the point number of this roof's point clouds


    """ ================================================================================================================ """
    """ Voxelization with direction correction """
    """ step 1: get main direction"""
    maindir = mdl_voxel.MainDirection(xyz)
    rect    = maindir.getMainDirection(img_resolution=256, pad_size=10)

    """ step 2: rotate point clouds"""
    M, xyz_rot = maindir.rotatePoints(rect=rect)

    """ step 3: voxelization"""
    voxelize = mdl_voxel.Voxelization(xyz_rot, grid_size=GRID_SIZE, thres_pcnt=0)
    voxels_center, voxels_idx = voxelize.getVoxels()

    # # visualize
    # visual.visual_2D(xyz_rot[:,0], xyz_rot[:,1])
    # visual.visual_3D(voxels_idx[:,0], voxels_idx[:,1], voxels_idx[:,2], labels=list(np.arange(0, len(voxels_idx), 1)),
    #                  IsSave='')


    """ ================================================================================================================ """
    """ Find Corner points """
    op_struct = mdl_fdStruct_v3_noOver.FindRoofStructure()

    """
    step 1: find topsurf
    """
    voxel_pd_topsurf = op_struct.find_topSurf(voxels_idx)


    """
    step 2: find and remove inner points
    """
    # speed-up，for a roof whose structure points have been calculated: directly import the relative file
    """txt file"""
    voxel_s_filename = StructP_FOLDER + os.path.basename(file_path).replace('.las', '')
    voxel_s_filename = './res/{}.csv'.format(voxel_s_filename + '_1struc')
    if StructP_FOLDER!='' and os.path.exists(voxel_s_filename):
        print(f"{voxel_s_filename} has existed.")
        voxel_pd_struct = pd.read_csv(voxel_s_filename)
    else:
        print(f"{voxel_s_filename} doesn't existed, is calculating...")
        max_search_radius_v1 = int(np.ceil((max(voxel_pd_topsurf['id_z']) + max_search_radius_param) / 2))  # TODO: hyper parmaeter
        voxel_pd_struct = op_struct.rm_innerPoint_v2(voxel_pd_topsurf, max_search_radius_v1)

    """
    step 3: find and remove isolated points
    """
    voxel_cand_pd  = op_struct.rm_isolateP_v2(voxel_pd_struct)


    """
    step 4: re-extract line points
    (more detailed explanation is shown in 'workflow_2_adjust.ipynb')
    """
    voxel_cand_line_pd, line_interior_idx = op_struct.save_lineP_v2(voxel_pd_topsurf, voxel_cand_pd)


    """
    step 5: Divide each line as several segments
    (more detailed explanation is shown in 'workflow_3_findCornerCandidate.ipynb')
    """

    """ step 5.0 get each line's corresponding id_z value"""
    cand_line_idx_pd = pd.DataFrame({'interior_idx': line_interior_idx}, dtype='object')
    cand_line_idx_pd[['id_z', 'l_a', 'l_b', 'l_c', 'l_len']] = cand_line_idx_pd.apply(
        lambda row: op_struct.sep_lG_and_get_lFunc(row, voxel_cand_line_pd), axis=1,
        result_type="expand")  # [l_a, l_b, l_c, l_len] is the line function's parameter

    """ 
    added part
    add time: 10th, Nov. 2022
    target: combine 0th & 1st layers' line
    """
    cand_line_idx_pd['id_z_bp'] = cand_line_idx_pd['id_z']
    cand_line_idx_pd.loc[cand_line_idx_pd['id_z_bp']==1, 'id_z'] = 0

    """ step 5.1 remove some strange lines """
    """
    ----------------------
    |     /              |
    |   /----------------
    | /  |
    ------
    """
    cand_line_idx_pd = op_struct.rm_abnormal_line(cand_line_idx_pd, voxel_cand_line_pd)

    """ step 5.2 group the lines' dataframe based on 'id_z', and get the iteration values (the unique value for id_z)"""
    cand_line_idx_pd = op_struct.get_cand_intersect_layer(cand_line_idx_pd)

    """ step 5.3 calculate intersections"""
    cand_line_idx_pd['intersectPs'] = cand_line_idx_pd.apply(
        lambda row: op_struct.calc_intersectP(row, cand_line_idx_pd, voxel_cand_line_pd), axis=1)

    """ step 5.4 calculate line intersections for getting line segments"""
    cand_line_idx_pd['new_lines'] = cand_line_idx_pd.apply(
        lambda row: op_struct.split_intersected_line(row, voxel_cand_line_pd), axis=1)
    # new_lines==-1 means all points on this line shouldn't be saved.
    cand_line_idx_pd = cand_line_idx_pd[cand_line_idx_pd['new_lines']!=-1]
    cand_line_idx_pd = cand_line_idx_pd.reset_index(drop=True)

    """
    step 6 Find final corners' x,y,z
    (more detailed explanation is shown in 'workflow_3_findCornerCandidate.ipynb')
    """
    """ step 6.1 Find final corners' corresponding voxel id_x, id_y, id_z"""
    LC_line_idx_pd = op_struct.divide_ngbr_sl_clusters(cand_line_idx_pd)
    fin_line_idx_pd = op_struct.extract_best_sl(LC_line_idx_pd, voxel_cand_line_pd)

    fin_line_idx_pd = fin_line_idx_pd[fin_line_idx_pd['isSave'] == True]
    fin_corner_idx = np.unique(sum(fin_line_idx_pd['new_lines'].values.tolist(), [])).tolist()
    # fin_corner_idx_pd includes: ['ofid', 'id_x', 'id_y', 'id_z']
    fin_corner_idx_pd = voxel_cand_line_pd[voxel_cand_line_pd['ofid'].isin(fin_corner_idx)]

    """ ================================================================================================================ """
    """ Extract coordinates, and re-rotate to geo-locations """
    op_geo_rerot = mdl_fdStruct_v3_noOver.FindCorners(fin_corner_idx_pd, voxels_center, M)
    fin_geo_corners = op_geo_rerot.get_Geo_Corners()


    """
    save result
    """
    save_name = SAVE_FOLDER + os.path.basename(file_path).replace('.las', '')

    """txt file for each step"""
    np.savetxt('./res/' + save_name + '_xyz_rot.txt', xyz_rot, delimiter=",", fmt="%.8f")
    read_data.save_dataframe('./res/{}.csv'.format(save_name + '_0topsurf'), voxel_pd_topsurf)
    read_data.save_dataframe('./res/{}.csv'.format(save_name + '_1struc'), voxel_pd_struct)
    read_data.save_dataframe('./res/{}.csv'.format(save_name + '_2cand'), voxel_cand_pd)
    read_data.save_dataframe('./res/{}.csv'.format(save_name + '_3structL'), voxel_cand_line_pd)

    read_data.save_dataframe('./res/{}.csv'.format(save_name + '_3-1finLine_idx'), fin_line_idx_pd)

    voxels_fin_line_idx = np.unique(sum(fin_line_idx_pd['interior_idx'].values.tolist(), [])).tolist()
    voxels_fin_line_pd = voxel_cand_line_pd[voxel_cand_line_pd['ofid'].isin(voxels_fin_line_idx)]
    read_data.save_dataframe('./res/{}.csv'.format(save_name + '_3-1finLine'), voxels_fin_line_pd)

    # if without edge
    read_data.save_dataframe('./res/{}.csv'.format(save_name + '_4finCor'), fin_corner_idx_pd)
    np.savetxt('./res/' + save_name + '_finCor_Geo.txt', fin_geo_corners, delimiter=",", fmt="%.8f")

    # visual graph
    # # Save topsurf res
    # visual.visual_3D(np.array(voxel_pd_topsurf['id_x']),
    #                  np.array(voxel_pd_topsurf['id_y']),
    #                  np.array(voxel_pd_topsurf['id_z']),
    #                  labels=list(voxel_pd_topsurf['ofid']), IsSave=save_name+'_0topsurf', IsonlySave=True)
    #
    # # Save structure  res
    # visual.visual_3D(np.array(voxel_pd_struct['id_x']),
    #                  np.array(voxel_pd_struct['id_y']),
    #                  np.array(voxel_pd_struct['id_z']),
    #                  labels=list(voxel_pd_struct['ofid']), IsSave=save_name + '_1struc', IsonlySave=True)
    #
    # # Save candidate point  res
    # visual.visual_3D(np.array(voxel_cand_pd['id_x']),
    #                  np.array(voxel_cand_pd['id_y']),
    #                  np.array(voxel_cand_pd['id_z']),
    #                  labels=list(voxel_cand_pd['ofid']), IsSave=save_name + '_2cand', IsonlySave=True)
    #
    # # Save structure line res
    # visual.visual_3D(np.array(voxel_cand_line_pd['id_x']),
    #                  np.array(voxel_cand_line_pd['id_y']),
    #                  np.array(voxel_cand_line_pd['id_z']),
    #                  labels = list(voxel_cand_line_pd['ofid']), IsSave=save_name+'_3structL', IsonlySave=True)
    #
    # # Save fin corner voxel res
    # visual.visual_3D(np.array(fin_corner_idx_pd['id_x']),
    #                  np.array(fin_corner_idx_pd['id_y']),
    #                  np.array(fin_corner_idx_pd['id_z']),
    #                  labels=list(fin_corner_idx_pd['ofid']), IsSave=save_name + '_4finCor', IsonlySave=True)
    #
    # # Save fin Geo-corners res
    # visual.visual_3D(fin_geo_corners[:, 0],
    #                  fin_geo_corners[:, 1],
    #                  fin_geo_corners[:, 2],
    #                  labels=list(np.arange(0, len(fin_geo_corners), 1)), IsSave=save_name + '_5finCor_Geo', IsonlySave=True)
