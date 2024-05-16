import numpy as np
from scipy import stats

import open3d as o3d
from shapely import geometry
from shapely import affinity
import alphashape
import cv2.cv2 as cv2


class MainDirection:
    def __init__(self, xyz):
        # for getoutlinePcd
        self.xyz = xyz

        # # for image_from_poly
        # self.resolution = img_resolution
        #
        # # for add_padding
        # self.pad_size = pad_size



    def getoutlinePcd(self):
        """
        using alphashape algorithm to get the outline of point cloud data for a building
        :return:
        """
        # points = xyz
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz)

        # calculate p_density
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        dd = np.zeros(len(self.xyz))  # num_points
        for j in range(len(self.xyz)):
            [_, idx_p, dis] = kdtree.search_knn_vector_3d(self.xyz[j], 2)
            dd[j] = dis[1]
        p_density_i = np.mean(np.sqrt(dd)) * 2
        alpha_shape = alphashape.alphashape(self.xyz[:, :2], p_density_i)
        # self.polygon = alpha_shape

        return alpha_shape

    def image_from_poly(self, polygon, resolution=256):
        img_matrix = np.zeros((resolution, resolution))

        if polygon.geom_type == 'MultiPolygon':
            # calculate each polygon's area, and get the geom with the max area
            geoms_area = [x.area for x in polygon.geoms]
            polygon = polygon.geoms[geoms_area.index(max(geoms_area))]


        tmp = polygon.exterior.coords
        all_points = np.asarray(tmp[:-1])
        x_max = all_points[:, 0].max()
        x_min = all_points[:, 0].min()
        y_max = all_points[:, 1].max()
        y_min = all_points[:, 1].min()

        x_diff = x_max - x_min
        y_diff = y_max - y_min

        if (x_diff >= y_diff):
            x_range = np.linspace(x_min, x_max, resolution + 1)
            y_range = np.linspace(y_min, y_min + x_diff, resolution + 1)
        else:
            y_range = np.linspace(y_min, y_max, resolution + 1)
            x_range = np.linspace(x_min, x_min + y_diff, resolution + 1)

        for i in range(len(x_range) - 1):
            for j in range(len(y_range) - 1):
                p1 = geometry.Point(x_range[i], y_range[j])
                p2 = geometry.Point(x_range[i], y_range[j + 1])
                p3 = geometry.Point(x_range[i + 1], y_range[j])
                p4 = geometry.Point(x_range[i + 1], y_range[j + 1])

                if polygon.contains(p1):
                    img_matrix[i, j] = 1
                    continue
                if polygon.contains(p2):
                    img_matrix[i, j] = 1
                    continue
                if polygon.contains(p3):
                    img_matrix[i, j] = 1
                    continue
                if polygon.contains(p4):
                    img_matrix[i, j] = 1
                    continue
                else:
                    img_matrix[i, j] = 0
        c = img_matrix.astype(int)
        d = np.asarray(c)
        d = d.astype('uint8')
        d = d * 255

        return d

    def add_padding(self, img, size):  # will always just add 2 pixels border around image, can be changed to whatever.
        row, col = img.shape[:2]
        bottom = img[row - 2:row, 0:col]
        mean = cv2.mean(bottom)[0]

        bordersize = size
        border = cv2.copyMakeBorder(
            img,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[0]
        )
        return border

    def getMainDirection(self,img_resolution=256, pad_size=10, return_alpha=False):
        """

        :return:
            rect:   minAreaRect, ((x,y), (x,y), angle)
                    e.g. ((135.64306640625, 132.6329345703125), (240.30145263671875, 247.28713989257812), 84.72611236572266)
        """
        a_shape = self.getoutlinePcd()
        alshp_img = self.image_from_poly(a_shape, img_resolution)
        alshp_img = self.add_padding(alshp_img, pad_size)
        # alshp_img_cp = cv2.cvtColor(alshp_img, cv2.COLOR_GRAY2BGR)

        alshp_cnts, hierarchy = cv2.findContours(alshp_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        cnt_i = 0
        for i in range(len(alshp_cnts)):
            if len(alshp_cnts[i]) > len(alshp_cnts[cnt_i]):
                cnt_i = i

        cnts = alshp_cnts[cnt_i]
        rect = cv2.minAreaRect(cnts)
        print('[MainDirection/getMainDirection()] :: minAreaRect: ', rect)
        # box = np.int0(cv2.boxPoints(rect))

        if return_alpha:
            return a_shape, rect
        else:
            return rect

    def rotatePoints(self, rect):
        xc, yc = np.mean(self.xyz[:, 0]), np.mean(self.xyz[:, 1])

        # TODO: double check this angle resetting part
        # reset rotate angle
        angle_radian = rect[-1] * np.pi / 180
        rotAng_cos = np.cos(angle_radian)
        rotAng_sin = np.sin(angle_radian)
        # TODO: END

        # rotate matrix
        # ==============================================================================================================
        # updated time: 01st, Nov. 2022
        # old rotate matrix: (has problem of translation)
        # M = [[rotAng_cos, -rotAng_sin, -1 * xc * rotAng_cos + yc * rotAng_sin + xc],
        #      [rotAng_sin, rotAng_cos, -1 * xc * rotAng_sin - yc * rotAng_cos + yc],
        #      [0, 0, 1]]
        # M = np.array(M)
        #
        # # get rotated points
        # xyz_rot = self.xyz[:, :3].copy()
        # xyz_rot[:, -1] = 0
        #
        # new version: modified the translation problem
        R = [[rotAng_cos, -rotAng_sin, 0],
             [rotAng_sin,  rotAng_cos, 0],
             [0, 0, 1]]
        R = np.array(R)
        T = [[1, 0, -xc],
             [0, 1, -yc],
             [0, 0, 1]]
        T = np.array(T)
        M = R @ T

        xyz_rot = self.xyz[:, :3].copy()
        xyz_rot[:, -1] = 1
        xyz_rot = M @ xyz_rot.T
        xyz_rot = xyz_rot.T

        # reset z values
        xyz_rot[:, -1] = self.xyz[:, 2]

        return M, xyz_rot

    def re_rotate_Points(self, xyz_rot, M):
        # get inverse M
        M_inv = np.linalg.pinv(M)

        xyz_re_rot = xyz_rot[:, :3].copy()
        xyz_re_rot[:, -1] = 1
        xyz_re_rot = M_inv @ xyz_re_rot.T
        xyz_re_rot = xyz_re_rot.T

        # reset z values
        xyz_re_rot[:, -1] = xyz_rot[:, 2]

        return xyz_re_rot


class Voxelization:
    def __init__(self, xyz_rot, grid_size=0.5, thres_pcnt=0):
        """
        :param xyz_rot:     rotated xyz points, numpy.array, [n,3]
        :param grid_size:   grid size
        :param thres_pcnt:  threshold, for removing grids with point number <= this threshold
        """
        self.xyz_rot = xyz_rot
        self.grid_size = grid_size
        self.thres_pcnt = thres_pcnt


    def getVoxels(self):
        """
        get voxels
        :return:
            h3d:           output shape: [n, m, p], voxel centers
            voxels_idx_N3: numpy.array [n,3]
        """
        # ==============================================================================================================
        # updated time: 01st, Nov. 2022
        # x_grid = np.arange(np.min(self.xyz_rot[:, 0]) - self.grid_size / 5,
        #                    np.max(self.xyz_rot[:, 0]) + self.grid_size,
        #                    self.grid_size)  # n
        # y_grid = np.arange(np.min(self.xyz_rot[:, 1]) - self.grid_size / 5,
        #                    np.max(self.xyz_rot[:, 1]) + self.grid_size,
        #                    self.grid_size)  # m
        # z_grid = np.arange(np.min(self.xyz_rot[:, 2]) - self.grid_size / 5,
        #                    np.max(self.xyz_rot[:, 2]) + self.grid_size,
        #                    self.grid_size)  # p

        x_grid = np.arange(np.min(self.xyz_rot[:, 0]) - self.grid_size / 2,
                           np.max(self.xyz_rot[:, 0]) + self.grid_size / 2,
                           self.grid_size)  # n
        y_grid = np.arange(np.min(self.xyz_rot[:, 1]) - self.grid_size / 2,
                           np.max(self.xyz_rot[:, 1]) + self.grid_size / 2,
                           self.grid_size)  # m
        z_grid = np.arange(np.min(self.xyz_rot[:, 2]) - self.grid_size / 2,
                           np.max(self.xyz_rot[:, 2]) + self.grid_size / 2,
                           self.grid_size)  # p

        # modified the last value
        x_grid[-1] = x_grid[-2] + (np.max(self.xyz_rot[:, 0]) - x_grid[-2]) * 2
        y_grid[-1] = y_grid[-2] + (np.max(self.xyz_rot[:, 1]) - y_grid[-2]) * 2
        z_grid[-1] = z_grid[-2] + (np.max(self.xyz_rot[:, 2]) - z_grid[-2]) * 2
        # ==============================================================================================================

        """calculate the number of points in each grid"""
        h3d = stats.binned_statistic_dd(self.xyz_rot, None, statistic='count', bins=[x_grid, y_grid, z_grid],
                                        expand_binnumbers=True)
        h3d = h3d.statistic  # output shape: [n, m, p]


        """find voxel id where point counts > thres"""
        voxels_idx = np.where(h3d > self.thres_pcnt)
        voxels_idx_N3 = np.vstack(voxels_idx).T

        return [x_grid, y_grid, z_grid], voxels_idx_N3




