"""
@File           : triangles.py
@Time           : 1/11/2023 4:33 PM
@Author         : Gefei Kong
@Description    : as below
--------------------------------------------------------------
get delaunay triangles of roof corner set
"""


import numpy as np

# from scipy.spatial import Delaunay
import triangle as tr # 3rd-party package

def triangluation_withcons(vertices, segs=None):
    if segs is None:
        data = dict(vertices=vertices)
        tri = tr.triangulate(data)
    else:
        data = dict(vertices=vertices, segments=segs)
        tri = tr.triangulate(data, 'pc')

    return tri







