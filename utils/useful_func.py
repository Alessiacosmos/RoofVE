"""
@File           : useful_func.py
@Time           : 1/11/2023 9:13 PM
@Author         : Gefei Kong
@Description    : as below
--------------------------------------------------------------

"""
import numpy as np
import shapely
from shapely.geometry import Polygon

def calc_intersect(base_poly, compare_poly):
    intersect_area = base_poly.intersection(compare_poly).area
    compare_area = compare_poly.area
    inter_percent = intersect_area / compare_area
    return inter_percent

def calc_area(poly_vertices, coords):
    poly_coords = coords[poly_vertices, :]
    poly = Polygon(poly_coords)
    return poly.area

def calc_intersect_vs(poly_v_compare, poly_v_base, coords):
    poly_compare_coords = coords[poly_v_compare, :]
    poly_compare = Polygon(poly_compare_coords)

    poly_base_coords = coords[poly_v_base, :]
    poly_base = Polygon(poly_base_coords)

    inter_p = calc_intersect(poly_base, poly_compare)

    return inter_p

def calc_face_normal(vertices_f):
    """
    using plane fitting to get the face normal
    reference link: https://stackoverflow.com/questions/64818203/how-to-find-the-coordinate-of-points-projection-on-a-planar-surface/64835893#64835893
    :param vertices_f: [n*3]
    :return:
    """
    # The adjusted plane crosses the centroid of the point collection
    centroid = np.mean(vertices_f, axis=0)

    # Use SVD to calculate the principal axes of the point collection
    # (eigenvectors) and their relative size (eigenvalues)
    _, values, vectors = np.linalg.svd(vertices_f - centroid)

    # Each singular value is paired with its vector and they are sorted from
    # largest to smallest value.
    # The adjusted plane plane must contain the eigenvectors corresponding to
    # the two largest eigenvalues. If only one eigenvector is different
    # from zero, then points are aligned and they don't define a plane.
    # if values[1] < 1e-6:
    #     raise ValueError("Points are aligned, can't define a plane")

    # So the plane normal is the eigenvector with the smallest eigenvalue
    normal = vectors[2]

    return normal
