import numpy as np
import pandas as pd

from sklearn.neighbors import KDTree
import itertools as it
import utils.module_voxelizaton as mdl_voxel
import utils.triangles_v2 as Tri

import shapely
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN

import utils.roof_graph_v3 as roof_G_v3
import utils.useful_func as ufunc

class FindRoofStructure:
    def __init__(self):
        pass

    def tag_innerPoint(self, fid, center_data, all_data):
        """
        seperate points around roof structure lines and innerpoints
        :param fid:
        :param center_data:
        :param all_data:
        :return:
        """
        # 0. if a point has been regarded as a non-structure pt: pass
        if center_data['isStruct'] == 'no':
            return 'no'

        # 1. get neighbors index (remove itself)
        ngbrs = list(center_data['neighbors'])

        # if only have itself in the ngbrs list, direct return and save this voxel
        # or the lowest layer should be always saved.
        if len(ngbrs) <= 1 or center_data['id_z'] == 0 or center_data['id_z'] == max(all_data['id_z']):
            return 'yes'
        # else, remove this voxel itself, and run following method
        ngbrs.remove(fid)

        # 2. get neighbors row datas
        ngbrs_row = all_data.loc[ngbrs, :]
        ngbrs_row['vx'] = ngbrs_row['id_x'] - center_data['id_x']
        ngbrs_row['vy'] = ngbrs_row['id_y'] - center_data['id_y']
        ngbrs_row['vz'] = ngbrs_row['id_z'] - center_data['id_z']

        # ==============================================================================================================
        # added part
        """added part: don't consider the points at the same layer"""
        ngbrs_non_samel = ngbrs_row[ngbrs_row['vz']!=0]
        ngbrs_non_samel = list(ngbrs_non_samel.index)
        # print('ngbrs: ', len(ngbrs), 'and ngbrs_not_at_same_layer: ', ngbrs_non_samel)

        """added part: for flat roof structure"""
        is_all_samel = False
        if len(ngbrs_non_samel) == 0: # if all neighbors of a pt are at the same layer: flat or isolated pt. ==> need further check.
            is_all_samel = True
        else: # else (all neighbors of a pt are NOT at the same layer): only consider neighbors from diff layers.
            ngbrs = ngbrs_non_samel
        # ==============================================================================================================


        num_180 = 0
        for ngbrComb_i in it.combinations(ngbrs, 2):  # # e.g. [1,2,3,4] -> [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
            # get each combination
            ngbrComb_i = list(ngbrComb_i)
            p1, p2 = ngbrs_row.loc[ngbrComb_i[0], :], ngbrs_row.loc[ngbrComb_i[1], :]
            v1, v2 = np.array(p1[['vx', 'vy', 'vz']]), np.array(p2[['vx', 'vy', 'vz']])

            # vector modules
            norm_v1 = np.sqrt(v1.dot(v1))
            norm_v2 = np.sqrt(v2.dot(v2))

            # dot product
            dot_v12 = v1.dot(v2)

            # calc cos
            cos_v12 = dot_v12 / (norm_v1 * norm_v2)

            # # get angle:
            # angle_r = np.arccos(cos_v12)
            # angle_d = angle_r * 180 / np.pi

            # 180 between v1(p1,center) and v2(p2,center)
            # 175 < angle_d <185: tolerance
            # if (angle_d < 190) and (angle_d > 170):
            # cos_v12==-1
            # if (angle_d < 190) and (angle_d > 170):
            if np.abs(cos_v12 - (-1)) < 6e-2:  # 6e-2: # 6e-2: tolerance to let the angle between 170-190
                if is_all_samel == False and p1['vz'] != 0 and p2['vz'] != 0:  # # p1, center, and p2 are from diff layers
                    return 'no'

                # ======================================================================================================
                # added part
                elif is_all_samel == True:
                    num_180 += 1
                    if num_180>3:   # if only having neighbors from the same layer and num(180_structure)>=4: return no
                        return 'no'
                # ======================================================================================================

        return 'yes'


    def fitLine(self, p1, p2, data_samel, thres_dis=0.4):
        """
        :param p1:
        :param p2:
        :param data_samel:
        :param thres_dis:
        :return:
        """
        # step 1: calculate line function
        # ax + by + c = 0
        # a = y2 - y1
        # b = x1 - x2
        # c = x2 * y1 - x1 * y2
        a = p2['id_y'] - p1['id_y']
        b = p1['id_x'] - p2['id_x']
        c = p2['id_x'] * p1['id_y'] - p1['id_x'] * p2['id_y']
        line_len = max(0.001, np.sqrt(a * a + b * b))

        # step 2: calculate the distance between the line and the point
        data_samel['dist'] = np.abs(a * data_samel['id_x'] + b * data_samel['id_y'] + c) / line_len
        # print('fitLine data_samel: ', np.array(data_samel[['ofid', 'id_x', 'id_y', 'id_z', 'dist']]))
        # step 3: find interior
        interior_set = data_samel.loc[data_samel['dist'] <= thres_dis, :]

        return interior_set['ofid'].tolist()


    def tag_line_point(self, center_data_cand, all_data):
        """
        find points on lines
        angle between two vectors: cos<a,b> = a·b/|a||b|
        :param fid:
        :param center_data:
        :param all_data:
        :return:
            a point index list which need to be removed
        """
        # extract center data from the topsurf data
        center_data = all_data[all_data['ofid'] == center_data_cand['ofid']].squeeze()
        fid = center_data['fid']

        # print('center data: \n', type(center_data_cand), '\n', type(center_data))
        # print('ofid: ', center_data_cand['ofid'])

        # 1. get neighbors index (remove itself)
        ngbrs = list(center_data['neighbors'])
        # print('first ngbrs: ', center_data['ofid'], ': ', ngbrs, len(ngbrs))

        # if only have itself in the ngbrs list, direct return and don't save this voxel -> it's an isolated voxel
        if len(ngbrs) <= 1:
            return []  # 'no'
        if 1 < len(ngbrs) <= 2:
            return [center_data['ofid']]  # 'yes'
        # else, remove this voxel itself, and run following method
        ngbrs.remove(fid)

        # 2. get neighbors row datas -> at the same layer
        ngbrs_row = all_data.loc[ngbrs, :]
        ngbrs_row['vx'] = ngbrs_row['id_x'] - center_data['id_x']
        ngbrs_row['vy'] = ngbrs_row['id_y'] - center_data['id_y']
        ngbrs_row['vz'] = ngbrs_row['id_z'] - center_data['id_z']

        ngbrs_row = ngbrs_row[ngbrs_row['vz'] == 0]  # -> at the same layer
        ngbrs = list(ngbrs_row.index)
        # print(ngbrs)
        if ngbrs_row.shape[0] == 0:  # save the points who don't have neighbors at the same layer
            return [center_data['ofid']]

        if ngbrs_row.shape[0] == 1:  # ext all points on the line organized by p and its unique neighbor point
            p1, p2 = center_data, ngbrs_row.squeeze()
            interior_idx_fin = self.fitLine(p1, p2, all_data[all_data['id_z'] == center_data['id_z']])
            return interior_idx_fin

        interior_idx_fin = []
        for ngbrComb_i in it.combinations(ngbrs, 2):  # # e.g. [1,2,3,4] -> [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
            # get each combination
            ngbrComb_i = list(ngbrComb_i)
            p1, p2 = ngbrs_row.loc[ngbrComb_i[0], :], ngbrs_row.loc[ngbrComb_i[1], :]
            v1, v2 = np.array(p1[['vx', 'vy', 'vz']]), np.array(p2[['vx', 'vy', 'vz']])

            # vector modules
            norm_v1 = np.sqrt(v1.dot(v1))
            norm_v2 = np.sqrt(v2.dot(v2))

            # dot product
            dot_v12 = v1.dot(v2)

            # calc cos
            cos_v12 = dot_v12 / (norm_v1 * norm_v2)
            # print(p1.fid, p2.fid, cos_v12)


            if ngbrs_row.shape[0] == 2:  # if only having two ngbrs at the same layer: check whether it can form a right angle: if yes: vertex, save it。
                if np.abs(cos_v12) < 1e-5:  # (close to 90 degree)
                    # ngbrs_idx = ngbrComb_i + [center_data['fid']]
                    # interior_idx_fin = all_data.loc[ngbrs_idx,'ofid'].tolist()
                    interior_idx_fin = [center_data['ofid']]
                    return interior_idx_fin


            if np.abs(cos_v12 - (-1)) < 1e-5:  # 6e-2: # 6e-2: 170-190 dgree
                # print('center: ', center_data['ofid'], 'p1, p2 and cos: ', p1.ofid, p2.ofid, cos_v12)
                # if p1['vz'] == p2['vz'] :#  # p1, center, p2来自同层
                #     return 'no'
                # if get 180 angle: fit line
                interior_idx = self.fitLine(p1, p2, all_data[all_data['id_z'] == center_data['id_z']])

                # save the longest line.
                if len(interior_idx) > len(interior_idx_fin):
                    interior_idx_fin = interior_idx

        return interior_idx_fin  # 'yes'


    def tag_line_point_v3(self, center_data_cand, all_data):
        """
        find points on lines v3
        angle between two vectors: cos<a,b> = a·b/|a||b|
        :param fid:
        :param center_data:
        :param all_data:
        :return:
            a point index list which need to be removed
        """
        # extract center data from the topsurf data
        center_data = all_data[all_data['ofid'] == center_data_cand['ofid']].squeeze()
        fid = center_data['fid']

        # print('center data: \n', type(center_data_cand), '\n', type(center_data))
        # print('ofid: ', center_data_cand['ofid'])

        # 1. get neighbors index
        ngbrs = list(center_data['neighbors'])
        # print('first ngbrs: ', center_data['ofid'], ': ', ngbrs, len(ngbrs))

        # if only have itself in the ngbrs list, direct return and don't save this voxel -> it's an isolated voxel
        if len(ngbrs) <= 1:
            return []  # 'no'
        # if 1 < len(ngbrs) <= 2:
        #     return [center_data['ofid']]  # 'yes'
        # # else, remove this voxel itself, and run following method

        # ==============================================================================================================
        # don't remove this voxel itself, and consider the line between the center and its each ngbr
        # updated time: 28th, Sept. 2022

        # 2. get neighbors row datas -> at the same layer
        ngbrs_row = all_data.loc[ngbrs, :]
        ngbrs_row['vx'] = ngbrs_row['id_x'] - center_data['id_x']
        ngbrs_row['vy'] = ngbrs_row['id_y'] - center_data['id_y']
        ngbrs_row['vz'] = ngbrs_row['id_z'] - center_data['id_z']

        ngbrs_row = ngbrs_row[ngbrs_row['vz'] == 0]  # -> at the same layer
        ngbrs = list(ngbrs_row.index)
        ngbrs.remove(fid)
        if len(ngbrs) < 1:
            return []  # 'no'

        interior_idx_fin = []
        for ngbrComb_i in range(len(ngbrs)):
            # get each combination
            p1, p2 = ngbrs_row.loc[fid, :], ngbrs_row.loc[ngbrs[ngbrComb_i], :]

            interior_idx = self.fitLine(p1, p2, all_data[all_data['id_z'] == center_data['id_z']])

            # save the longest line.
            if len(interior_idx) > len(interior_idx_fin):
                interior_idx_fin = interior_idx

        return interior_idx_fin  # 'yes'
        # ==============================================================================================================


    def find_topSurf(self, voxels_idx):
        voxel_pd = pd.DataFrame(voxels_idx, columns=['id_x', 'id_y', 'id_z'])
        voxel_pd = pd.concat([pd.DataFrame(voxel_pd.index, columns=['ofid']), voxel_pd], axis=1)
        voxel_pd_bottom = voxel_pd[voxel_pd['id_z'] == 0]
        voxel_pd_nonbtm = voxel_pd[voxel_pd['id_z'] != 0]
        voxel_pd_nonbtm = voxel_pd_nonbtm.drop_duplicates(subset=['id_x', 'id_y'], keep='last', inplace=False)
        voxel_pd_topsurf = pd.concat([voxel_pd_bottom, voxel_pd_nonbtm])
        voxel_pd_topsurf = voxel_pd_topsurf.sort_values(by=['ofid'])

        return voxel_pd_topsurf



    def rm_innerPoint_v2(self, voxel_pd_topsurf, max_search_radius):
        """
        rm innerpoints of roof surfaces, only keep points around roof structure lines/vertices
        :param voxel_pd_topsurf:    topsurface roof point clouds
        :param max_search_radius:   max_search_radius
        :return:
               voxel_pd_struct:     roof point clouds without obvious inner points
        """

        # =============================================================================================
        # create the basic neighbors info.
        voxel_pd_struct = voxel_pd_topsurf[['ofid', 'id_x', 'id_y', 'id_z']].copy()
        voxel_struct_idx = np.array(voxel_pd_struct[['ofid', 'id_x', 'id_y', 'id_z']])
        kd_cheby = KDTree(voxel_struct_idx[:, 1:], metric='chebyshev') # chebyshev distance

        # init for first round
        voxel_pd_struct = pd.DataFrame(voxel_struct_idx, columns=['ofid', 'id_x', 'id_y', 'id_z'])
        voxel_pd_struct = pd.concat([pd.DataFrame(voxel_pd_struct.index, columns=['fid']), voxel_pd_struct], axis=1)
        # =============================================================================================
        for i in range(max_search_radius):
            sradius_i = i + 1

            # col_isStruct = voxel_pd_struct['isStruct']
            voxel_ngbrs_cheby = kd_cheby.query_radius(voxel_struct_idx[:, 1:],
                                                      r=sradius_i)  # build neighbor index list
            # print(voxel_ngbrs_cheby.shape, voxel_pd_struct.shape)
            voxel_pd_struct['neighbors'] = voxel_ngbrs_cheby
            # voxel_pd_struct['isStruct']  = col_isStruct
            if sradius_i == 1:
                voxel_pd_struct['isStruct'] = 'yes'

            voxel_pd_struct['isStruct'] = voxel_pd_struct.apply(
                lambda row: self.tag_innerPoint(row['fid'], row, voxel_pd_struct), axis=1)

            print('[FindRoofStructure/rm_innerPoint_v2()] :: sr={} - struct point number: '.format(i),
                  voxel_pd_struct[voxel_pd_struct['isStruct'] == 'yes'].shape)

        voxel_pd_struct = voxel_pd_struct[voxel_pd_struct['isStruct'] == 'yes']

        return voxel_pd_struct


    def rm_isolateP_v2(self, voxel_pd_struct):
        """attention: cannot rm isolated points at top and bottom layers"""
        print('is using rm_isolateP_v2...')

        max_z = max(voxel_pd_struct['id_z'])
        min_z = min(voxel_pd_struct['id_z'])
        """以 2 为半径(chebyshev distance)建立KD tree"""
        # cand means candidate
        voxels_cand_idx = np.array(voxel_pd_struct[['ofid', 'id_x', 'id_y', 'id_z']])
        kd_cand = KDTree(voxels_cand_idx[:, 1:], metric='chebyshev')
        voxel_cand_ngbrs = kd_cand.query_radius(voxels_cand_idx[:, 1:], r=2)  # build neighbor index list
        # voxel_cand_ngbrs_cnt = np.array([len(_) for _ in voxel_cand_ngbrs])  # count neighbor's number

        voxel_cand_pd = pd.DataFrame(voxels_cand_idx, columns=['ofid', 'id_x', 'id_y', 'id_z'])
        # voxel_cand_pd['ngbr_cnt_r2'] = voxel_cand_ngbrs_cnt
        voxel_cand_pd = pd.concat([pd.DataFrame(voxel_cand_pd.index, columns=['fid']), voxel_cand_pd], axis=1)

        # ==============================================================================================================
        # new updated time: 28th, Sept, 2022
        # remove candidate points which has <=1 neighbors when searching radius = 2 '''at the same layer '''
        # in these structure points

        def cnt_ngbr_samel(center_data, all_data):
            fid = center_data['fid']

            # 1. get neighbors index (remove itself)
            ngbrs = list(center_data['ngbr_r2'])
            # print('first ngbrs: ', center_data['ofid'], ': ', ngbrs, len(ngbrs))
            # if only have itself in the ngbrs list, direct return and don't save this voxel -> it's an isolated voxel
            if len(ngbrs) <= 1:
                return len(ngbrs)
            # else, remove this voxel itself, and run following method
            ngbrs.remove(fid)

            # 2. get neighbors row datas -> at the same layer
            ngbrs_row = all_data.loc[ngbrs, :]
            ngbrs_row = ngbrs_row[ngbrs_row['id_z'] == center_data['id_z']]  # -> at the same layer
            ngbrs = list(ngbrs_row.index)
            return len(ngbrs)+1

        voxel_cand_pd['ngbr_r2'] = voxel_cand_ngbrs
        voxel_cand_pd['ngbr_cnt_r2'] = voxel_cand_pd.apply(lambda row: cnt_ngbr_samel(row, voxel_cand_pd), axis=1)
        voxel_cand_pd = voxel_cand_pd[(voxel_cand_pd['ngbr_cnt_r2'] > 1) |
                                      ((voxel_cand_pd['id_z']==max_z))
                                      | (voxel_cand_pd['id_z']==min_z) ]  # >1：because the point itself is included in the neighbor points.

        # ==============================================================================================================

        """re-create KD-tree with radius=1 (chebyshev distance), to get the right neighbor poitns"""
        # cand means candidate
        voxels_cand_idx2 = np.array(voxel_cand_pd[['ofid', 'id_x', 'id_y', 'id_z']])
        kd_cand2 = KDTree(voxels_cand_idx2[:, 1:], metric='chebyshev')
        voxel_cand_ngbrs2 = kd_cand2.query_radius(voxels_cand_idx2[:, 1:], r=1)  # build neighbor index list

        voxel_cand_pd = pd.DataFrame(voxels_cand_idx2, columns=['ofid', 'id_x', 'id_y', 'id_z'])
        voxel_cand_pd['neighbors'] = voxel_cand_ngbrs2

        voxel_cand_pd = pd.concat([pd.DataFrame(voxel_cand_pd.index, columns=['fid']), voxel_cand_pd], axis=1)

        return voxel_cand_pd


    def save_lineP_v2(self, voxel_pd_topsurf, voxel_cand_pd):
        """
        find points on roof structure lines
        :param voxel_pd_topsurf:
        :param voxel_cand_pd:
        :return:
            voxel_cand_line_pd: pd.Dataframe ['ofid', 'id_x', 'id_y', 'id_z'] from voxel_pd_topsurf
            line_interios_idx:  list [k, n],  k = the number of structure lines,
                                              n = the point number in this line, which is different for each line
        """
        """For the original topsurf data: create KD-tree with radius=1(chebyshev distance) """
        # cand means candidate
        voxels_topsurf_idx = np.array(voxel_pd_topsurf[['ofid', 'id_x', 'id_y', 'id_z']])
        kd_topsurf = KDTree(voxels_topsurf_idx[:, 1:], metric='chebyshev')
        voxels_topsurf_ngbrs = kd_topsurf.query_radius(voxels_topsurf_idx[:, 1:], r=1)  # build neighbor index list ---> updated time: 11th, Nov. 2022

        voxel_pd_topsurf_ngbr = pd.DataFrame(voxels_topsurf_idx, columns=['ofid', 'id_x', 'id_y', 'id_z'])
        voxel_pd_topsurf_ngbr['neighbors'] = voxels_topsurf_ngbrs
        voxel_pd_topsurf_ngbr = pd.concat(
            [pd.DataFrame(voxel_pd_topsurf_ngbr.index, columns=['fid']), voxel_pd_topsurf_ngbr], axis=1)

        # ==============================================================================================================
        # updated to tag_line_point_v3
        # updated time: 28th, Sept. 2022
        """For voxel_cand_pd_new: create KD-tree with radius=1(chebyshev distance)"""
        voxel_cand_pd['interior_idx'] = voxel_cand_pd.apply(
            lambda row: self.tag_line_point_v3(row, voxel_pd_topsurf_ngbr), axis=1)
        # ==============================================================================================================

        """Find points on roof structure lines"""
        # voxel_cand_idx = np.unique(np.concatenate(voxel_cand_pd['interior_idx'].tolist())).astype('int')
        # voxel_cand_line_pd = voxel_pd_topsurf[voxel_pd_topsurf['ofid'].isin(voxel_cand_idx)]

        # ==============================================================================================================
        # updated part
        # new updated time: 28th, Sept, 2022
        tmp_interior_idx = voxel_cand_pd['interior_idx'].apply(lambda row: ','.join(map(str, row)))
        tmp_interior_idx_save = tmp_interior_idx.value_counts()
        tmp_interior_idx_save = tmp_interior_idx_save[
            (tmp_interior_idx_save != 1) & (tmp_interior_idx_save.index != '')].index

        line_interios_idx = [list(map(int, x.split(','))) for x in tmp_interior_idx_save]
        tmp_interior_idx_save_unq = np.unique(np.array(sum(line_interios_idx, []), dtype=int))
        voxel_cand_line_pd = voxel_pd_topsurf[voxel_pd_topsurf['ofid'].isin(tmp_interior_idx_save_unq)]

        # ==============================================================================================================

        return voxel_cand_line_pd, line_interios_idx


    def calc_overlap_ratio_iter(self, a_line, row_lidx, curr_line_p, is_rm2=False):
        """
        calculate the overlap ratio between a line's point set and another line
        :param a_line:
        :param row_lidx:
        :param curr_line_p:
        :param is_rm2:
        :return:
        """
        a_line_lidx = a_line.name[-1]

        # 如果是平行的直线，直接不计算，返回0
        if a_line_lidx == row_lidx:
            return 0
        # 如果是非平行的直线，计算重叠比率
        else:
            a, b, c = a_line['l_a'], a_line['l_b'], a_line['l_c']
            # 1. 计算直线row上的所有点curr_line_p 到 另一条直线 a_line 的距离
            # dist = (Ax+By+C)/√(A^2+B^2), 在这里，因为l_a, l_b, l_c均已经单位化过，无需再 /√(A^2+B^2)
            curr_line_p['dist'] = np.abs(a * curr_line_p['id_x'] + b * curr_line_p['id_y'] + c)
            # print('a_line:\n', a_line)
            # print('dist:\n', curr_line_p)

            # 2. 计算overlap_ratio
            # 2.1 距离在 1 切比雪夫距离 内的 点数
            thres = np.sqrt(2)
            num_in_1cheby = curr_line_p[curr_line_p['dist'] < thres].shape[0]
            num_total = curr_line_p.shape[0]
            if is_rm2:
                num_total += 2
            overlap_ratio = num_in_1cheby / num_total

            return overlap_ratio

    def calc_overlap_ratio(self, row, all_lines, all_p_pd):
        """
        calculate the overlap ratio between a line's point set and all other non-parallel lines
        :param row:
        :param all_lines:
        :param all_p_pd:
        :return:
        """
        # 计算该直线与所有直线的重叠率
        # 1. 取出该直线上对应点所有的xyz信息
        curr_line_idxs = row['interior_idx']
        # 两个端点不被考虑
        is_rm2 = False
        # ==============================================================================================================
        # updated time: 10th, Nov. 2022
        if len(curr_line_idxs) > 2:
            is_rm2 = True
            curr_line_idxs = curr_line_idxs[1:-1]
        # ==============================================================================================================
        curr_line_p = all_p_pd[all_p_pd['ofid'].isin(curr_line_idxs)]

        # 2. 计算这些点到每条直线的距离
        row_lidx = row.name[-1]

        row_overlap_ratio = all_lines.apply(
            lambda a_line: self.calc_overlap_ratio_iter(a_line, row_lidx, curr_line_p, is_rm2), axis=1)
        row_overlap_ratio = row_overlap_ratio.to_numpy()
        # print(row_overlap_ratio)

        return row_overlap_ratio

    def rm_abnormal_line(self, cand_line_idx_pd, voxel_cand_line_pd):
        """
        Updated time: 8th, Nov. 2022
        remove abnormal lines like following:
        ----------------------
        |     /              |
        |   /----------------
        | /  |
        ------
        :param cand_line_idx_pd:
        :param voxel_cand_line_pd:
        :return:
        """

        cand_line_idx_pd['isAbnormal'] = False

        cand_line_idx_pd_rm_abnormal = []
        for key, line_gb in cand_line_idx_pd.groupby(by=['id_z']):  # , 'l_a', 'l_b'
            line_gb['l_a'] = line_gb['l_a'].round(6)
            line_gb['l_b'] = line_gb['l_b'].round(6)

            line_gb['idx_str'] = line_gb['l_a'].astype('str') + ',' + line_gb['l_b'].astype('str')
            line_gb = line_gb.set_index(['idx_str'], append=True)

            # # create matrix saving overlap ratios， ij <--> ji are corresponded
            # overlap_matrix = np.zeros(shape=(len(all_index_p1), len(all_index_p1)))
            # # return each line's overlap ratios
            overlap_matrix = line_gb.apply(lambda row: self.calc_overlap_ratio(row, line_gb, voxel_cand_line_pd),
                                             axis=1)
            overlap_matrix = np.array(overlap_matrix.tolist())
            # print(overlap_matrix.round(3))

            overlap_matrix_up = np.triu(overlap_matrix, 1)  # upper triangular matrix, [n, n]
            overlap_matrix_lo = np.tril(overlap_matrix, -1).T  # lower triangular matrix, [n, n], T: let ij<-->ji in this matrix corresponds to them in tri_u
            overlap_matrix_ul = np.stack([overlap_matrix_up, overlap_matrix_lo], axis=0)  # shape=[2, n, n]

            """
            # rm abnormal lines
            """
            # Step 1: find places [0, i, j] or [1, i, j] > 0.33
            # e.g. overlap_33_idx =
            # array([[0, 0, 0, 1, 1, 1, 2, 3],
            #        [1, 2, 3, 2, 3, 4, 3, 4]], dtype=int64) row1: i,row2: j
            overlap_33_idx = np.where(overlap_matrix_ul > 0.33)[1:]  # shape: [2, x]
            overlap_33_idx_unq = np.unique(overlap_33_idx, axis=1)  # remove duplicated elements by columns
            overlap_33_idx_unq = (overlap_33_idx_unq[0], overlap_33_idx_unq[1])
            # print('overlap_33_idx_unq: \n', overlap_33_idx_unq, overlap_33_idx_unq[0].shape)

            # Step 2: find the corresponding overlap ratios of all [:,i,j]
            # e.g. o_all =
            # [[0.10714286 0.03571429 0.04761905 0.04761905]
            #  [0.33333333 0.33333333 0.33333333 0.33333333]]
            o_ij = overlap_matrix_ul[0][overlap_33_idx_unq]
            o_ji = overlap_matrix_ul[1][overlap_33_idx_unq]
            o_all = np.vstack((o_ij, o_ji))
            # print('o_all:\n', o_all, o_all.shape)

            # Step 3: check which is larger: [0,i,j] or [1,i,j]
            # e.g., o_larger_idx =
            # array([0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], dtype=int64), where 0 means i, 1 means j
            # for o_larger_idx = 0: it means that o(li-->lj) > o(lj-->li), (the higher overlap ratios of li ON lj: li should be the abnormal line,rm li
            # 即: isAbnormal_i =True
            o_larger_idx = o_all.argmax(axis=0)

            # Step 4: find corresponding i or j
            # 4.1 add index
            # e.g.
            # array([[0, 1, 2, 3, 4, 5, 6, 7],
            #        [1, 0, 1, 1, 1, 1, 0, 1]], dtype=int64)
            o_larger_idx = np.vstack((np.arange(overlap_33_idx_unq[0].shape[0]), o_larger_idx))
            o_larger_idx = (o_larger_idx[0], o_larger_idx[1])  # 制作成索引的格式
            # print('o_larger_idx:\n', o_larger_idx)
            # 4.2 get (i,j) of indexes
            overlap_33_idx_unq_T = np.array(overlap_33_idx_unq).T
            # print(overlap_33_idx_unq_T)
            # 4.3 return the row idx of lines need to be rm.
            abnormal_l_idx = overlap_33_idx_unq_T[o_larger_idx].tolist()
            # print('abnormal_l_idx: \n', abnormal_l_idx)
            # Step 5
            line_gb.iloc[abnormal_l_idx, -1] = True  # -1 means 'isAbnormal'
            line_gb.index = line_gb.index.droplevel(-1)

            cand_line_idx_pd_rm_abnormal.append(line_gb)

        cand_line_idx_pd_rm_abnormal = pd.concat(cand_line_idx_pd_rm_abnormal, axis=0)
        cand_line_idx_pd_rm_abnormal = cand_line_idx_pd_rm_abnormal[cand_line_idx_pd_rm_abnormal['isAbnormal'] == False]

        return cand_line_idx_pd_rm_abnormal


    def get_Line_Function(self, p1, p2):
        """
        # calculate line function
        # ax + by + c = 0
        # a = y2 - y1
        # b = x1 - x2
        # c = x2 * y1 - x1 * y2
        :param p1:  point 1 [x,y,z]
        :param p2:  point 2 [x,y,z]
        :return:
               a,b,c (normalized), line_len
        """

        a = p2[1] - p1[1]  # p2['id_y'] - p1['id_y']
        b = p1[0] - p2[0]  # p1['id_x'] - p2['id_x']
        c = p2[0] * p1[1] - p1[0] * p2[1]  # p2['id_x'] * p1['id_y'] - p1['id_x'] * p2['id_y']
        line_len = max(0.001, np.sqrt(a * a + b * b))
        a, b, c = a / line_len, b / line_len, c / line_len  # normalize

        return a, b, c, line_len

    def get_line_cross_point(self, line1, line2, sp_idx, ep_idx, all_voxel_pd):
        # x1y1x2y2
        a0, b0, c0 = line1[0], line1[1], line1[2]
        a1, b1, c1 = line2[0], line2[1], line2[2]
        D = a0 * b1 - a1 * b0
        if D == 0:
            return 'None'
        x = (b0 * c1 - b1 * c0) / D
        y = (a1 * c0 - a0 * c1) / D
        # print(x, y)

        # check whether the intersection is out of the segments of uppder lines: if yes, this intersection doesn't exist actually.
        sp = all_voxel_pd[all_voxel_pd['ofid'] == sp_idx]
        ep = all_voxel_pd[all_voxel_pd['ofid'] == ep_idx]

        # length of segments
        len_segment = np.sqrt(np.square(sp['id_x'].values[0] - ep['id_x'].values[0]) +
                              np.square(sp['id_y'].values[0] - ep['id_y'].values[0]))
        # length of segments (p, start_p_of_segment) and (p, end_p_of_segment)
        len_crp_sp = np.sqrt(np.square(x - sp['id_x'].values[0]) + np.square(y - sp['id_y'].values[0]))
        len_crp_ep = np.sqrt(np.square(x - ep['id_x'].values[0]) + np.square(y - ep['id_y'].values[0]))
        # get the longer segment
        len_crp_seg = max(len_crp_sp, len_crp_ep)

        # consider the following situation: intersection is out of segment [sp, ep]
        # *---------------*  #(intersection(x,y))
        # sp              ep
        # tag this situation as "No" intersection
        if len_crp_seg > len_segment:
            return 'None'

        return [x, y]

    def sep_lG_and_get_lFunc(self, line, all_p_pd):
        """
        separate line to groups based on id_z
        and get each line's function
        :param line:     pd.Series, line[0]: list, (n) n = the number of this line's points, which is different for each line
        :param all_p_pd: voxel_cand_line_pd all points in all lines, including ['ofid', 'id_x', 'id_y', 'id_z']
        :return:
               line_z:   the layer id of this line（id_z)
               l_a:      A: Ax+By+C=0
               l_b:      B: Ax+By+C=0
               l_c:      C: Ax+By+C=0
               l_len:    length of line segments
        """

        line_include_idx = line['interior_idx']

        """1. separate line to groups based on id_z"""
        line_p0_idx = line_include_idx[0]
        line_z = all_p_pd.loc[all_p_pd['ofid'] == line_p0_idx, 'id_z'].squeeze()

        """2. get each line's function"""
        p1 = all_p_pd.loc[all_p_pd['ofid'] == line_include_idx[0], ['id_x', 'id_y']].values  # e.g. [[1,2]]
        p2 = all_p_pd.loc[all_p_pd['ofid'] == line_include_idx[-1], ['id_x', 'id_y']].values
        l_a, l_b, l_c, l_len = self.get_Line_Function(p1[0], p2[0])

        return line_z, l_a, l_b, l_c, l_len


    def get_cand_intersect_layer(self, cand_line_idx_pd):
        """
        group the lines' dataframe based on 'id_z', and get the iteration values (the unique value for id_z)
        :param cand_line_idx_pd:
        :return:
        """
        # get 'key' for iteration
        gb_key = np.unique(cand_line_idx_pd['id_z'])  # np.array [n, ]
        gb_key[::-1].sort()  # descending

        cand_line_idx_pd['cand_intersect_z'] = '-1'
        # cand_line_idx_pd['cand_intersect_z'] = cand_line_idx_pd['cand_intersect_z'].astype('object')
        # example of gb_key = [8,6,0] -> [top_z, middle_z, bottom_z]
        for i in range(1, len(gb_key)):
            cand_intersect_z = []
            if i == 0:  # top
                cand_intersect_z = []
            elif i == (len(gb_key)) - 1:  # bottom
                cand_intersect_z = gb_key[:-1] # consider all upper layers, including middle and top
            else:  # middle
                cand_intersect_z = [gb_key[0]] # only top
            cand_intersect_z = ','.join(map(str, cand_intersect_z))
            cand_line_idx_pd['cand_intersect_z'].loc[cand_line_idx_pd['id_z'] == gb_key[i]] = cand_intersect_z

        return cand_line_idx_pd


    def calc_intersectP(self, line, all_line_pd, all_voxel_pd):
        """
        calc intersections between lines
        :param line:
        :param all_line_pd:
        :param all_voxel_pd:
        :return:
        """

        if line['cand_intersect_z'] == '-1':  # 顶层，不用算交点
            # interP_toplayer = self.calc_intersectP_toplyr(line, all_line_pd, all_voxel_pd)
            return []

        """get current line's function parameter"""
        a_curr, b_curr, c_curr = line['l_a'], line['l_b'], line['l_c']

        """
        get point indexes in lines which need to calculate the intersection with the current line
        # e.g. a pd.Dataframe
        # [l_a, l_b, l_c, l_len] is the line function's parameter
        #   interior_idx, id_z, l_a, l_b, l_c, l_len, cand_intersect_z
        # 2 [......],     0,    0,   1,   -14, 10,    6
        # 4 [......],     0,    1,   -5,  7,   22,    6
        """
        cand_intersect_z = np.array(line['cand_intersect_z'].split(','), dtype=float)
        upper_lines = all_line_pd[all_line_pd['id_z'].isin(cand_intersect_z)]

        """
        calc the intersections between a line and all upper_lines (multi lines)
        reference: https://blog.csdn.net/ONE_SIX_MIX/article/details/107413321
        """
        upper_lines['intersectP'] = upper_lines.apply(
            lambda row: self.get_line_cross_point([a_curr, b_curr, c_curr],
                                                  [row['l_a'], row['l_b'], row['l_c']],
                                                  row['interior_idx'][0],
                                                  row['interior_idx'][-1],
                                                  all_voxel_pd), axis=1)

        intersectPs = upper_lines.loc[upper_lines['intersectP'] != 'None', 'intersectP'].tolist()

        # print(line['interior_idx'][0], ' upper_lines: \n', upper_lines)

        return intersectPs


    def split_intersected_line(self, row, all_p_pd):
        """
        split the line which intersected with other layers' lines
        :param row:
        :param all_p_pd:
        :return:
        """

        """don't consider vertices from the top layer here"""
        if row['cand_intersect_z'] == '-1':
            return [
                [row['interior_idx'][0], row['interior_idx'][-1]]]  # line start and end point (ofid), [[start, end]]

        """check whether existing intersections: if no: the line doesn't intersect with other lines, directly return its [start, end]"""
        if len(row['intersectPs']) == 0:
            return [
                [row['interior_idx'][0], row['interior_idx'][-1]]]  # line start and end point (ofid), [[start, end]]

        """0. get all intersect points"""
        intersectPs = np.array(row['intersectPs'])

        if len(intersectPs.shape) == 1:  # sovling special case of only one intersection: [x,y]
            intersectPs = np.array([intersectPs])

        """1. get the current line's all segments"""
        curr_line_idxs = row['interior_idx']
        curr_segs = all_p_pd[all_p_pd['ofid'].isin(curr_line_idxs)]

        """2. get the (x,y) info. of each segment's mid point and the length of each segment """
        curr_segs_midP = (curr_segs.iloc[:-1, 1:3] + curr_segs.iloc[1:, 1:3].values) / 2
        curr_segs_midP.columns = ['mid_id_x', 'mid_id_y']

        curr_segs_len = np.sqrt(np.square(curr_segs.iloc[:-1, 1] - curr_segs.iloc[1:, 1].values) +
                                np.square(curr_segs.iloc[:-1, 2] - curr_segs.iloc[1:, 2].values))
        curr_segs_len = curr_segs_len.to_frame('seg_len')

        """
        3. get ofid_end info., and create 2 new columns:
        1. ofid_end,
        2: seg_seid: = [ofid, ofid_end], to target segment start and end point's idx
        """
        curr_segs_ofid_end = curr_segs.iloc[1:, 0].values

        curr_segs = pd.concat([curr_segs.iloc[:-1, :], curr_segs_midP, curr_segs_len], axis=1)

        curr_segs.insert(1, 'ofid_end', curr_segs_ofid_end)

        curr_segs_seid = np.array([curr_segs['ofid'].values] + [curr_segs['ofid_end'].values]).T.tolist()
        curr_segs.insert(2, 'seg_seid', curr_segs_seid)

        """3. calculate the distance between mid_p and intersectPs, to get the segs which prepare to be deleted """
        need_del_seg = []
        for interP_i in range(intersectPs.shape[0]):
            intersectP_i = intersectPs[interP_i]  # [x, y]
            curr_segs['dist'] = np.sqrt(np.square(curr_segs['mid_id_x'] - intersectP_i[0]) +
                                        np.square(curr_segs['mid_id_y'] - intersectP_i[1]))

            # only rm(i.e., merge) this segment when min(dist(intersection，middle_segment)) <= 1/2 of this line segment
            # because when this dist > 1/2 of this line segment，this intersection is out of the segment
            # ---------------   |
            # ---------------   # <==intersection
            # ---------------   |
            seg_cand_del = curr_segs[curr_segs['dist'] == min(curr_segs['dist'])]

            # if this intersection is at the middle of two segs
            #     | <== endpoint of segs, as well as an intersection
            # *---*---*
            # this intersection is NOT out of the segment, consider it
            if seg_cand_del.shape[0] > 1:
                need_del_seg.append(
                    curr_segs.loc[curr_segs['dist'] == min(curr_segs['dist']), ['ofid', 'ofid_end']].values.tolist())
            else:
                # consider min(dist) <= 1/2 of this line segment
                if min(curr_segs['dist']) <= seg_cand_del['seg_len'].values / 2:
                    need_del_seg.append(
                        curr_segs.loc[
                            curr_segs['dist'] == min(curr_segs['dist']), ['ofid', 'ofid_end']].values.tolist())
            # print(seg_cand_del)
            # print(curr_segs)

        """4. get segmented line"""
        """ curr_segs: ['ofid', 'ofid_end', 'seg_seid', 'id_x', 'id_y', 'id_z', 'mid_id_x', 'mid_id_y', 'seg_len', 'dist'] """

        # preprocessing: convert need_del_seg format
        # [[[s11,e11]], [[s21,e21],[s22,e22]]] -> [[s11,e11],[s21,e21],[s22,e22]]
        # final shape: n*2
        if len(need_del_seg)!=0: # need_del_seg!=[]
            need_del_seg = np.unique(np.array(sum(need_del_seg, []), dtype=int), axis=0)
            # sort need_del_seg
            need_del_seg = need_del_seg[need_del_seg[:, 0].argsort()].tolist()
            # 4.1. rm
            new_segs = curr_segs[~curr_segs['seg_seid'].isin(need_del_seg)]
        else:
            new_segs = curr_segs

        # 4.2. merge all points on a segment
        # 4.2.1
        new_segs['isContinue'] = new_segs.iloc[:-1, 1] - new_segs.iloc[1:, 0].values

        # 4.2.2: get index of all row whose isContinue!=0 and !=NaN(last row)
        new_segs = new_segs.reset_index(drop=True)  # reset a index because the original index = ofid (not continue)
        new_seg_split_idx = new_segs[new_segs['isContinue'] != 0].index.tolist()

        # 4.2.3: get new segs
        new_segs_list = []
        for i in range(len(new_seg_split_idx)):
            split_idx_i = new_seg_split_idx[i]

            if i == 0:
                seg_split_i = [new_segs.loc[0, 'ofid'], new_segs.loc[split_idx_i, 'ofid_end']]
            else:
                split_idx_pre = new_seg_split_idx[i - 1]
                seg_split_i = [new_segs.loc[split_idx_pre + 1, 'ofid'], new_segs.loc[split_idx_i, 'ofid_end']]

            new_segs_list.append(seg_split_i)

        if len(new_segs_list)==0: # consider special case: there is no points saved on this segment.
            return -1

        return new_segs_list


    def create_vector(self, row, voxel_cand_line_pd, thresRO):
        """
        e.g.
           p11                     p12
        l1 *----------*---------->*
        l2                  *----------*------------------->*
                           p21                            p22
                            |~~~~~| <== overlapped part
        :param row:
        :param voxel_cand_line_pd:
        :return:
        """

        if row['next_new_lines'] == -1:  # last line, whose relationship has been considered in the previous line.
            return -1

        # l1 （本行）
        se_idx_row = [row['new_lines'][0][0], row['new_lines'][-1][-1]]
        p11 = voxel_cand_line_pd.loc[voxel_cand_line_pd['ofid'] == se_idx_row[0], ['id_x', 'id_y']].values[
            0]  # .tolist()
        p12 = voxel_cand_line_pd.loc[voxel_cand_line_pd['ofid'] == se_idx_row[1], ['id_x', 'id_y']].values[
            0]  # .tolist()
        # vec1 = [p12['id_x'] - p11['id_x'], p12['id_y'] - p11['id_y']]
        vec1 = p12 - p11
        vec1_len = np.sqrt(np.sum(np.square(vec1)))

        # l2 （下一行）
        se_idx_next_row = [row['next_new_lines'][0][0], row['next_new_lines'][-1][-1]]
        p21 = voxel_cand_line_pd.loc[voxel_cand_line_pd['ofid'] == se_idx_next_row[0], ['id_x', 'id_y']].values[
            0]  # .tolist()
        p22 = voxel_cand_line_pd.loc[voxel_cand_line_pd['ofid'] == se_idx_next_row[1], ['id_x', 'id_y']].values[
            0]  # .tolist()
        # vec2s = [p21['id_x'] - p11['id_x'],p21['id_y'] - p11['id_y']] # l1起点 与 l2起点 构成的向量
        # vec2e = [p22['id_x'] - p11['id_x'],p22['id_y'] - p11['id_y']] # l1起点 与 l2终点 构成的向量
        vec2s = p21 - p11
        vec2e = p22 - p11

        # 计算 vec2s 和 vec2e 落在 vec1 上 分别的ratio
        r_s = np.dot(vec1, vec2s) / vec1_len ** 2
        r_e = np.dot(vec1, vec2e) / vec1_len ** 2

        # print(vec1, vec2s, vec2e, r_s, r_e)

        # check ratio
        # this line can be the potential split point only when there is a ratio in [0,1]
        if 0 <= r_s <= 1 or 0 <= r_e <= 1:
            # print('0<=r_s<=1 or 0<=r_e<=1')
            if 0 <= r_s <= 1 and 0 <= r_e <= 1:  # corresponding to the following situation: don't split
                # *--------------*
                #        *----*
                # print('0<=r_s<=1 and 0<=r_e<=1')
                return -1  # don't split
            # else:  # calc the overlap ratio
                # if r_s > 0 and r_e > 0:
                #     # situation 1:
                #     # *----------->*
                #     #          *-------*
                #     r_s, r_e = 1 - r_s, 1 - r_e
                # # else:
                # # situation 2:
                # #     *----------->*
                # # *-------*
                #
                # if 0 < r_s < thresRO or 0 < r_e < thresRO:  # overlap ratio<0.2: split
                #     # print('0 < r_s < 0.2 or 0 < r_s < 0.2')
                #     return 1
                # else:
                #     return -1  # don't split

        else:  # all ratios <0 or >1, this line must be a split position
            if (r_s < 0 and r_e < 0) or (r_s > 1 and r_e > 1):
                # print('all >0 or <1')
                return 1  # split

        return -1  # default: don't split


    def divide_ngbr_sl_clusters_p1(self, cand_line_idx_pd):
        """
        cluster neighbored segmentline of each parallel line group in each layer
        e.g. In a layer (with the same id_z)
        p1         p2          p3                       p4
        *----------*-----------*------------------------* Ax+By+C1 = 0 - LC1
         *---------*-----------*----------*               Ax+By+C2 = 0 - LC1


        *----------*-----------*------------------------* Ax+By+C3 = 0 - LC2

        This function aims to find LCi

        :param cand_line_idx_pd:
        :return:
        """
        """init"""
        cand_line_idx_pd['LC_idx'] = -1

        """iteration： Step 1: group based on layer, and get LC of each layer Ci (num_C = k)"""
        # Rule 4 -> groupby['id_z', 'l_a', 'l_b']->
        # Rule 5 -> the result of this for-loop(LC_idx)
        LC_line_idx_pdlist = []
        for key, line_gb in cand_line_idx_pd.groupby(by=['id_z', 'l_a', 'l_b']):
            # print('======================is solving group {}==============================='.format(key))
            """Step 2: get LC"""
            num_l_ingb = line_gb.shape[0]  # number of lines in LC
            """if num_lines(LC)<=1,directly return LC_idx=0 because there will be only 1 group"""
            if num_l_ingb <= 1:
                line_gb['LC_idx'] = 0
            else:
                """preprocessing: sort based on l_c"""
                line_gb = line_gb.sort_values(by=['l_c'])

                """Step 2.1: find multiple LC in Ci (number of LCs = k_LC)"""
                dist_inCi = -line_gb.iloc[:-1, 4].values + line_gb.iloc[1:, 4].values  # next_line['l_c'] - previous_line['l_c']
                # prl_cdist_np   = np.array(line_gb['cdist'].values.tolist())
                idx_prl_clu = np.where(dist_inCi > 1)[0].tolist()  # threshold==1 neighbor.
                idx_prl_clu.append(line_gb.shape[0] - 1)  # the iloc_idx of last line
                # print('dist_inCi: ', dist_inCi, idx_prl_clu, len(idx_prl_clu))

                for k_LC in range(len(idx_prl_clu)):
                    if k_LC == 0:  # the first LC: [0: idx_prl_clu[0]]
                        sLC_idx, eLC_idx = 0, idx_prl_clu[k_LC] + 1
                    else:
                        sLC_idx, eLC_idx = idx_prl_clu[k_LC - 1] + 1, idx_prl_clu[k_LC] + 1
                    line_gb.iloc[sLC_idx: eLC_idx, -1] = k_LC  # -1: 'LC_idx'

            LC_line_idx_pdlist.append(line_gb)
            # print(line_gb)

        LC_line_idx_pd = pd.concat(LC_line_idx_pdlist, axis=0)

        return LC_line_idx_pd

    def divide_ngbr_sl_clusters_p2(self, cand_line_idx_pd, voxel_cand_line_pd, thresRO):
        """
        (this rule is not used in the final version of the method)
        cluster neighbored segmentline of each parallel line group in each layer - part 2
        consider overlap ratio
        *----------*-----------*                        A1x+B1y+C1 = 0     - LC1
                            *------------------------*  A1x+B1y+(C1-1) = 0 - LC2
        :param cand_line_idx_pd:
        :return:
        """
        """init"""
        # cand_line_idx_pd['LC_idx'] = -1
        cand_line_idx_pd['LC_idx_p2'] = -1

        """iteration： Step 1: group based on layer, and get LC_p2 of each layer LCi (num_LC = k_LC)"""
        LC_line_idx_pdlist = []
        # Rule 4 & 5 -> groupby['id_z', 'l_a', 'l_b', 'LC_idx]
        # Rule 6     -> 'LC_idx_p2'
        for key, line_gb in cand_line_idx_pd.groupby(by=['id_z', 'l_a', 'l_b', 'LC_idx']):
            # print('======================is solving group {}==============================='.format(key))
            """Step 2: check overlap ratios """
            # print(line_gb, line_gb.shape)
            num_l_ingb = line_gb.shape[0]  # number of lines in LC
            """if num_lines(LC)<=1,directly return LC_idx=0 because there will be only 1 group"""
            if num_l_ingb <= 1:
                line_gb['LC_idx_p2'] = 0
                LC_line_idx_pdlist.append(line_gb)
            else:
                """preprocessing: sort based on l_c"""
                line_gb = line_gb.sort_values(by=['l_c'])

                """Step 2.1: find multiple cluster LC_p2 in LCi (number of LC_p2s = k_LC_p2)"""
                """Step 2.1.1: create vector<p1, p2>: [start, end] point of each new line"""
                next_new_lines = line_gb['new_lines'].values.tolist()
                next_new_lines = next_new_lines[1:]  # 2nd row -> last row
                next_new_lines.append(-1)  # keep the same size for merging
                line_gb['next_new_lines'] = next_new_lines

                is_further_divide = line_gb.apply(lambda row: self.create_vector(row, voxel_cand_line_pd, thresRO=thresRO), axis=1)
                idx_p2_clu = np.where(is_further_divide == 1)[0].tolist()
                idx_p2_clu.append(line_gb.shape[0] - 1)


                for k_LC_p2 in range(len(idx_p2_clu)):
                    if k_LC_p2 == 0:  # first cluster: [0: idx_prl_clu[0]]
                        sLC_idx, eLC_idx = 0, idx_p2_clu[k_LC_p2] + 1
                    else:
                        sLC_idx, eLC_idx = idx_p2_clu[k_LC_p2 - 1] + 1, idx_p2_clu[k_LC_p2] + 1
                    line_gb.iloc[sLC_idx: eLC_idx, -2] = k_LC_p2  # -2: 'LC_idx_p2', -1: new_lines

                # print(line_gb, '\n', is_further_divide)

                LC_line_idx_pdlist.append(line_gb.iloc[:, :-1])
            # print(line_gb)

        LC_line_idx_pd = pd.concat(LC_line_idx_pdlist, axis=0)

        return LC_line_idx_pd


    def divide_ngbr_sl_clusters(self, cand_line_idx_pd):
        """
        consider cluster rules for grouping.
        :param cand_line_idx_pd:
        :param voxel_cand_line_pd:
        :return:
        """
        LC_line_idx_pd = self.divide_ngbr_sl_clusters_p1(cand_line_idx_pd)
        # rm a rule: the new column is used to replace this rule's return.
        # LC_line_idx_pd = self.divide_ngbr_sl_clusters_p2(LC_line_idx_pd, voxel_cand_line_pd, thresRO)
        LC_line_idx_pd['LC_idx_p2'] = 0

        return LC_line_idx_pd


    def extract_best_sl(self, LC_line_idx_pd, voxel_cand_line_pd):
        """
        There are a lot of parallel lines in the extracted candidate lines, which may represent the same corner
        This function aims to contact these parallel lines which represent the same structure corner.

        e.g. In a layer (with the same id_z)
        p1         p2          p3                       p4
        *----------*-----------*------------------------* Ax+By+C1 = 0 - LC1
         *---------*-----------*----------*               Ax+By+C2 = 0 - LC1


        *----------*-----------*------------------------* Ax+By+C3 = 0 - LC2

        In this function, for the first cluster LC1, the line 'Ax+By+C1=0' will be saved and 'Ax+By+C2=0' will be removed

        :param LC_line_idx_pd:
        :return:
        """
        """init"""
        LC_line_idx_pd['isSave'] = True

        fin_line_idx_pdlist = []
        """Iteration： Step 1&2: group by layer, LC, and line params (A,B). line_LC are lines in a LCi"""
        # ==============================================================================================================
        # modified time: 04th, Nov. 2022
        # original version forgets the cluster_part2
        # for key, line_LC in LC_line_idx_pd.groupby(by=['id_z', 'l_a', 'l_b', 'LC_idx']):
        #
        # new version: --> add 'LC_idx_p2' for grouping
        # for key, line_LC in LC_line_idx_pd.groupby(by=['id_z', 'l_a', 'l_b', 'LC_idx', 'LC_idx_p2']):
        print("[extract_best_sl()]::'LC_idx_p2' is not covered in groupby.")
        for key, line_LC in LC_line_idx_pd.groupby(by=['id_z', 'l_a', 'l_b', 'LC_idx']):
        # ==============================================================================================================
            # print('======================is solving group {}==============================='.format(key))

            """Step 3: process for each LCi"""
            """Step 3&4: check the number of lines in each LCi. Only merge when num_line>1"""
            num_l_inLC = line_LC.shape[0]  # the number of lines in each LCi

            if num_l_inLC > 1:
                """preprocessing: sort based on l_c"""
                line_LC = line_LC.sort_values(by=['l_c'])

                allline_inLC_idx = [True for i in range(num_l_inLC)]
                """Step 5&6: check num_l_ingb is odd or even"""
                if num_l_inLC % 2 != 0:  # odd
                    savedline_idx = int((num_l_inLC - 1) / 2)
                else:  # even
                    # 6.1. find two lines closest the avg center.
                    cand_lc_idx = [int(np.floor((num_l_inLC - 1) / 2)), int(np.ceil((num_l_inLC - 1) / 2))]

                    cand_lc_segn = []
                    cand_lc_len = []
                    for i in range(2):
                        cand_lc_i = line_LC.iloc[cand_lc_idx[i]]
                        # 6.2. get number of segments in these two candidate lines
                        cand_lc_segn.append(len(cand_lc_i['new_lines']))

                        # 6.3. calc the segment length of these two candidate lines
                        cand_lc_i_se_ofid = [cand_lc_i['new_lines'][0][0], cand_lc_i['new_lines'][-1][-1]]
                        cand_lc_i_se = voxel_cand_line_pd[voxel_cand_line_pd['ofid'].isin(cand_lc_i_se_ofid)]
                        cand_lc_i_len = np.sqrt(np.square(cand_lc_i_se['id_x'].iloc[0] - cand_lc_i_se['id_x'].iloc[1]) +
                                                np.square(cand_lc_i_se['id_y'].iloc[0] - cand_lc_i_se['id_y'].iloc[1]))
                        cand_lc_len.append(cand_lc_i_len)

                    if cand_lc_segn[0] != cand_lc_segn[1]:  # their number of segments are not equal
                        # 6.2. save the line with less segments
                        savedline_idx = cand_lc_segn.index(min(cand_lc_segn))  # 0 or 1
                    else:  # else
                        # 6.3. save the longer line
                        savedline_idx = cand_lc_len.index(max(cand_lc_len))  # 0 or 1

                    savedline_idx = cand_lc_idx[savedline_idx]  # get the real idx of the line in this group

                allline_inLC_idx[savedline_idx] = False
                line_LC.iloc[allline_inLC_idx, -1] = False  # iloc[:, -1]: isSave

            fin_line_idx_pdlist.append(line_LC)
            # print(line_LC)

        fin_line_idx_pd = pd.concat(fin_line_idx_pdlist, axis=0)

        return fin_line_idx_pd



class FindFaces_by_Delaunay:
    def __init__(self):
        pass

    def get_triangels(self, fin_line_idx_pd, fin_corner_idx_pd, fin_corner_idx):
        """
        Triangulation with constraints.
        constraint：new_lines
        :param fin_corner_idx_pd:
        :param fin_corner_idx:
        :return:
        """
        vertices = fin_corner_idx_pd[['id_x', 'id_y', 'id_z']].values
        edges_base = np.array(sum(fin_line_idx_pd["new_lines"].values.tolist(), []))

        edges_base_s, edges_base_e = edges_base[:, 0], edges_base[:, 1]
        edges_base_reidx = np.array([np.where(fin_corner_idx == edges_base_s[:, None])[-1],
                                     np.where(fin_corner_idx == edges_base_e[:, None])[-1]]).T  # index从0开始

        dtri = Tri.triangluation_withcons(vertices[:, :2], segs=edges_base_reidx)  # 投影到xy平面
        dtri_tri = dtri['triangles']
        vertices_dtri_added = dtri["vertices"][len(vertices):, :]

        return vertices, dtri_tri, vertices_dtri_added


    def add_corners_based_dtri(self, fin_corner_idx, vertices_dtri_added, voxels_idx, vertices, dtri_tri):
        vertices_addlist = []
        for i in range(len(vertices_dtri_added)):
            v_add_i_x, v_add_i_y = int(vertices_dtri_added[i, 0]), int(vertices_dtri_added[i, 1])  # int(xy)
            v_add_i_idx = np.where((voxels_idx[:, 0] == v_add_i_x) & (voxels_idx[:, 1] == v_add_i_y))[0]
            if len(v_add_i_idx) != 0:  # if the added vertices locates in voxel set, directly add.
                v_add_i_idx = v_add_i_idx[0]
                v_add_i = list(voxels_idx[v_add_i_idx])
                vertices_addlist.append(v_add_i)
            else:  # else: move it to the closest voxel
                v_dist = np.sqrt(np.square(voxels_idx[:, 0] - v_add_i_x) +
                                 np.square(voxels_idx[:, 1] - v_add_i_y))
                v_add_i_idx = np.where(v_dist == np.min(v_dist))[0][0]
                v_add_i = list(voxels_idx[v_add_i_idx])
                # check whether this point is in existing vertex set: if not: add; else: del this tri_vertices,and update the corresponding triangles' index of this vertex
                if v_add_i_idx not in fin_corner_idx:
                    vertices_addlist.append(v_add_i)
                else:
                    v_exist_idx = np.where((vertices[:, 0] == v_add_i[0]) &
                                           (vertices[:, 1] == v_add_i[1]) &
                                           (vertices[:, 2] == v_add_i[2]))[0][0]
                    dtri_tri[dtri_tri == (len(vertices) + i - 1)] = v_exist_idx
                    # print(v_exist_idx)
            # print(v_add_i_idx, v_add_i)

            fin_corner_idx.append(v_add_i_idx)
        vertices = np.vstack([vertices, vertices_addlist])

        return vertices, fin_corner_idx


    def rm_tris_not_intersect_with_voxelshape(self, ashape_voxel, vertices, dtri_tri):
        """
        rm unexpected generated triangles (not in alphashape(voxels))
        :param ashape_voxel:
        :param vertices:
        :param dtri_tri:
        :return:
        """
        saved_tri = []
        saved_tri_coords = []
        for i in range(len(dtri_tri)):
            tr_coord_i = vertices[dtri_tri[i]]  # only extract x and y
            # tr_coord_i = coords_i[:,:2]
            tr_coord_i_xy = np.vstack([tr_coord_i[:, :2], tr_coord_i[0, :2]])  # create enclose shape
            tr_xy_poly = Polygon(tr_coord_i_xy)
            tr_i_inter_percent = ufunc.calc_intersect(ashape_voxel, tr_xy_poly)
            # print(dtri_tri[i], tr_i_inter_percent)

            if tr_i_inter_percent > 0.7:  # two standard deviation
                saved_tri.append(dtri_tri[i])
                saved_tri_coords.append(tr_coord_i)

        saved_tri = np.array(saved_tri)

        return saved_tri, saved_tri_coords


    def get_tris_normals(self, saved_tri: np.ndarray, saved_tri_coords: list) -> np.ndarray:
        """
        calc tris normals
        :param saved_tri:
        :param saved_tri_coords:
        :return:
        """
        saved_tri_normal = []
        for stri_i in range(len(saved_tri)):
            tri_normal = ufunc.calc_face_normal(saved_tri_coords[stri_i])
            # saved_tri_normal.append(tri_normal if tri_normal[0] >= 0 else tri_normal * (-1))
            if tri_normal[0] > 0:
                tri_normal = tri_normal
            elif tri_normal[0] < 0:
                tri_normal = tri_normal * (-1)
            else:  # =0,solve -0
                if tri_normal[1] < 0:
                    tri_normal = tri_normal * (-1)
                if tri_normal[1] == 0 and tri_normal[2] < 0:
                    tri_normal = tri_normal * (-1)
            saved_tri_normal.append(tri_normal)
            # print(f"normal {stri_i} for {saved_tri[stri_i]}: ", saved_tri_normal[stri_i])

        saved_tri_normal = np.array(saved_tri_normal)

        return saved_tri_normal

    def merge_tris(self, saved_tri: np.ndarray, saved_tri_normal: np.ndarray, vertices: np.ndarray,
                   db_eps: float=0.3) -> list:
        """"""
        clustering_normal = DBSCAN(eps=db_eps, min_samples=1).fit(saved_tri_normal)  # 经验
        saved_tri_normallbl = clustering_normal.labels_
        saved_tri_lbl_num = np.unique(saved_tri_normallbl)

        """merge: merge faces by finding max cycle"""
        fin_faces = []
        for lbl_i in range(len(saved_tri_lbl_num)):
            tri_group_i_idx = np.where(saved_tri_normallbl == lbl_i)[0]
            tri_group_i = saved_tri[tri_group_i_idx, :]  # all triangle faces, np.array(), shape=[n, 3]
            tri_gi_edges = self.poly_2_edges_v2(polys=tri_group_i, keep_duplicated_edge=False)
            # find all max cycles of this triangle group
            gi_cycles, cycle_matrix, cnt_cycle = roof_G_v3.find_all_cycles(tri_gi_edges)
            gi_max_cycles = roof_G_v3.find_max_cycles_v2(vertices, gi_cycles)
            fin_faces.append(gi_max_cycles)

        fin_faces = sum(fin_faces, [])

        return fin_faces

    def get_fin_faces(self, fin_line_idx_pd, fin_corner_idx_pd, fin_corner_idx, ashape_voxel, voxels_idx) -> list:
        # step 1: triangulation
        vertices, dtri_tri, vertices_dtri_added = self.get_triangels(fin_line_idx_pd, fin_corner_idx_pd, fin_corner_idx)

        if len(vertices_dtri_added)>0: # there might some updated vertices during delaunay triangulation
            print("Delaunay returns additional vertices, need add them.")
            vertices, fin_corner_idx = self.add_corners_based_dtri(fin_corner_idx, vertices_dtri_added,
                                                                   voxels_idx, vertices, dtri_tri)

        # step 2: rm unexpected generatedtriangles
        saved_tri, saved_tri_coords = self.rm_tris_not_intersect_with_voxelshape(ashape_voxel, vertices, dtri_tri)

        # step 3: calc face normals
        saved_tri_normal = self.get_tris_normals(saved_tri, saved_tri_coords)

        # step 4: cluster and merge triangles based on normals
        fin_faces = self.merge_tris(saved_tri, saved_tri_normal, vertices)

        return fin_faces, fin_corner_idx # return updated fin_corner_idx


    def get_fin_edges(self, faces: list) -> np.ndarray:
        fin_edges = self.poly_2_edges(faces)

        return fin_edges


    def poly_2_edges(self, polys: list) -> np.ndarray:
        """
        ext all edges of multi polygons
        :param polys:        the vertex list of multi polygons， e.g.: [[0,1,2,3,4], [1,5,7]]
        :return:
        """
        is_cycle_format = True if polys[0][0] == polys[0][-1] else False
        # print(f"whether the input poly is formatted as a cycle: {is_cycle_format}")

        edges = []
        for p in polys:
            if is_cycle_format:
                edges_i = np.vstack([p[:-1], p[1:]]).T
            else:
                edge_end = np.roll(p, 1)
                edges_i = np.vstack([p, edge_end]).T
            edges.append(edges_i.tolist())
        edges = sorted(sum(edges, []))
        edges = np.array(edges)
        edges = np.unique(edges, axis=0)
        return edges

    def poly_2_edges_v2(self, polys: list, keep_duplicated_edge=True) -> np.ndarray:
        """
        ext all edges of multi polygons
        :param polys:        the vertex list of multi polygons， e.g.: [[0,1,2,3,4], [1,5,7]]
        :return:
        """
        is_cycle_format = True if polys[0][0] == polys[0][-1] else False
        # print(f"whether the input poly is formatted as a cycle: {is_cycle_format}")

        edges = []
        for p in polys:
            if is_cycle_format:
                edges_i = np.vstack([p[:-1], p[1:]]).T
            else:
                edge_end = np.roll(p, 1)
                edges_i = np.vstack([p, edge_end]).T
            edges.append(edges_i.tolist())

        edges = sorted(sum(edges, []))
        edges = np.array(edges)
        edges = np.sort(edges, axis=1)
        if not keep_duplicated_edge:
            rm_idx = np.where(pd.DataFrame(edges).duplicated(keep=False))[0]
            edges = np.delete(edges, rm_idx, axis=0)
        else:
            edges = np.unique(edges, axis=0)
        return edges



class FindCorners:
    def __init__(self, fin_corner_idx_pd, voxels_center, M):
        self.fin_corner_idx = fin_corner_idx_pd
        self.voxels_center  = voxels_center
        self.M = M

    def get_geoCoord_each_row(self, row, allvoxels_center):
        geo_x = (allvoxels_center[0][row['id_x']] + allvoxels_center[0][row['id_x'] + 1]) / 2
        geo_y = (allvoxels_center[1][row['id_y']] + allvoxels_center[1][row['id_y'] + 1]) / 2
        geo_z = (allvoxels_center[2][row['id_z']] + allvoxels_center[2][row['id_z'] + 1]) / 2

        return [geo_x, geo_y, geo_z]

    def get_geoCoord_each_row_v2(self, row, allvoxels_center, xyz_rot):
        geo_xbox = [allvoxels_center[0][row['id_x']], allvoxels_center[0][row['id_x'] + 1]]
        geo_ybox = [allvoxels_center[1][row['id_y']], allvoxels_center[1][row['id_y'] + 1]]
        geo_zbox = [allvoxels_center[2][row['id_z']], allvoxels_center[2][row['id_z'] + 1]]

        # find point clouds in this voxel
        lb = [geo_xbox[0], geo_ybox[0], geo_zbox[0]]  # lower bound of xyz
        ub = [geo_xbox[1], geo_ybox[1], geo_zbox[1]]  # upper bound of xyz
        xyz_rot_inbox_idx = np.all((lb <= xyz_rot) & (xyz_rot <= ub), axis=1)
        xyz_rot_inbox = xyz_rot[xyz_rot_inbox_idx]

        # get Centroid of these points
        geo_center = np.mean(xyz_rot_inbox, axis=0)

        return geo_center

    def get_geoCoord(self):
        self.fin_corner_idx['geo_coords'] = self.fin_corner_idx.apply(
            lambda row: self.get_geoCoord_each_row(row, self.voxels_center), axis=1)
        fin_corners = np.array(self.fin_corner_idx['geo_coords'].to_list())

        return fin_corners

    def get_geoCoord_v2(self, xyz_rot):
        self.fin_corner_idx['geo_coords'] = self.fin_corner_idx.apply(
            lambda row: self.get_geoCoord_each_row_v2(row, self.voxels_center, xyz_rot), axis=1)
        fin_corners = np.array(self.fin_corner_idx['geo_coords'].to_list())

        return fin_corners


    def get_rerot_geoCoord(self, fin_corners, M):
        maindir = mdl_voxel.MainDirection(None)
        fin_corners_re_rotate = maindir.re_rotate_Points(fin_corners, M)

        return fin_corners_re_rotate

    def get_Geo_Corners(self):
        fin_corners = self.get_geoCoord()
        fin_geo_corners_re_rotate = self.get_rerot_geoCoord(fin_corners, self.M)

        return fin_geo_corners_re_rotate

    def get_Geo_Corners_v2(self, xyz_rot):
        fin_corners = self.get_geoCoord_v2(xyz_rot)
        fin_geo_corners_re_rotate = self.get_rerot_geoCoord(fin_corners, self.M)

        return fin_geo_corners_re_rotate

