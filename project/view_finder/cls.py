# -*- coding: utf-8 -*-

# AUTHOR: YAN FU
# DATE: 22/01/2018
# NAME: /view_finder/cls.py
# Description: This file contains objects which required for this project 'view finder'.

import pandas as pd


class Surface(object):
    """
    Description: A single surface that defined (by user) by a series cooridations. Meshing will be performed within this
    object. The process upon instantiate are: (1) obtain surface coordiantes, temperature and type; (2) transform
    """
    def __init__(self):
        pass


class Surfaces(object):
    """
    Description:
    """
    def __init__(self):
        pass


class Points(object):
    """
    Description: This object serves as collection of each individual points on a surface, which theoretically should be
    the centroid of each sub-surface shape. This object, Point, should be able to store all neccessary information for
    radiation calculation: (a) location coordinates; (b) direction vectors; (c) types; and (d) temperatures.
    """
    def __init__(self):
        """"""
        self.data_all = pd.DataFrame({
            'location': (0,0,0)
        })

    def add_surface(self, Surface):
        pass
