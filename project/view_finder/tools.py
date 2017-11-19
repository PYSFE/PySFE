# -*- coding: utf-8 -*-
import copy
import re

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import project.func.transformations as mat


def parse_inputs_from_text(raw_input_text_str):

    # with open(input_file_dir) as input_file:
    #     input_text = input_file.read()

    # remove spaces
    input_text = raw_input_text_str.replace(" ", "")

    # break raw input text to a list
    input_text = re.split(';|\r|\n', input_text)

    # delete comments (anything followed by #)
    input_text = [v.split("#")[0] for v in input_text]

    # delete empty entries
    input_text = [v for v in input_text if v]

    # for each entry in the list (input_text), break up i.e. ["variable_name=1+1"] to [["variable_name"], ["1+1"]]
    input_text = [v.split("=") for v in input_text]

    # analyse for each individual element input and create a list of library
    input_text_dict = list()
    for each_entry in input_text:
        if '*' in each_entry[0]:
            each_entry = str(each_entry[0]).replace('*', '')
            input_text_dict.append({'keyword': each_entry})
        else:
            input_text_dict[len(input_text_dict)-1][str(each_entry[0])] = eval(str(each_entry[1]))

    return input_text


def find_plane_norm(plane_vertices_sequence):
    a = np.asarray(plane_vertices_sequence)

    if len(a) >= 3:
        return np.cross(a[1] - a[0], a[2] - a[0])
    else:
        return 0


def find_plane_function(plane_vertices_sequence):
    """
    DESCRIPTION:
        Calculate the plane which all vertex in vertices are on it. Plane is represented in a, b, c and d, which
        a x + b y + c z + d == 0
    PARAMETERS:
        vertices    {list}      a list with every individual cell describing a coordinate in (x, y, z)
    """

    # compute plane equation
    v = np.asarray(plane_vertices_sequence)

    a, b, c = tuple(np.cross(v[1, :] - v[0, :], v[2, :] - v[0, :]))
    d = -np.sum(np.array([a, b, c]) * v[0, :])

    return a, b, c, d


def find_angle_between_vectors_3d(vector1, vector2):
    a = np.dot(vector1, vector2)
    b = np.linalg.norm(np.cross(vector1, vector2))
    theta = np.arctan2(b, a)  # angle between the two vectors in clock-wise direction
    theta = np.pi - theta  # convert the angle to anti-clockwise (right hand rule)
    return theta


def find_rotated_vertices_3d(vertices_3d, rotation_axis, rotation_angles, decimals=12):
    """\
    This function rotates input vertices 'vertices_3d' about axis vector 'rotation_axis' for an angle in radians 'rotation_angles'.
    :param vertices_3d:     a list of vertex coordinates in format [x, y, z].
    :param rotation_angles: A number represents an angle in radiant.
    :param rotation_axis:   A vector [x, y, z] represents rotation axis.
    :param decimals:        Results accuracy, by default is 12.
    :return vertices_2d:
    :return vertices_3d_rotated:
    :return z_axis_value:
    """

    vertices_mat = np.matrix(vertices_3d)
    row_size = np.size(vertices_mat, axis=0)
    zeros_column_mat = np.ones((row_size, 1), dtype=float)
    vertices_mat = np.append(vertices_mat, zeros_column_mat, axis=1)

    rotation_matrix = np.matrix(mat.rotation_matrix(rotation_angles, rotation_axis))

    vertices_mat = vertices_mat.transpose()
    vertices_3d_rotated = rotation_matrix * vertices_mat
    vertices_3d_rotated = vertices_3d_rotated[0:(np.size(vertices_3d_rotated, 0) - 1), :]
    vertices_3d_rotated = vertices_3d_rotated.transpose()
    vertices_3d_rotated = np.asarray(vertices_3d_rotated).round(decimals=decimals)

    return vertices_3d_rotated


def plot_poly_axes3d(file_name, verts, magnitudes=None,
                     title=None, limits=None, figure=None, figure_ax=None, width_scale=1):
    '''
    PARAMETERS:
        file_name           {string, -}         File name with extension i.e. '.png'. Plot will be saved according to
                                                this name.
        poly_coordinates    {list float, -}     It contains a list of polygon. i.e. [subpoly1, subpoly2, ...]
                                                Each 'subpoly' is defined by a sequence of coordinates.
                                                i.e. [[x1, y1, z1], [x2, y2, z2], ...]
        magnitudes_scaled          {list float, -}     A list of number.
    REMARKS:
        1. if figure_ax is not None, then it has to be an Axes3D object.
    '''

    # x = [coordinate[0] for coordinate in poly_vertices]
    # y = [coordinate[1] for coordinate in poly_vertices]
    # z = [coordinate[2] for coordinate in poly_vertices]


    # instantiate figure
    if not figure_ax:
        fig = plt.figure(figsize=(width_scale * 8.3, width_scale * 8.3/1.3333333333))
        fig_ax = Axes3D(fig)
    else:
        fig = figure
        fig_ax = figure_ax

    # set title
    if title:
        fig_ax.set_title(title)

    # set labels
    fig_ax.set_xlabel('x')
    fig_ax.set_ylabel('y')
    fig_ax.set_zlabel('z')

    # set limits
    if limits:
        fig_ax.set_xlim(limits[0])
        fig_ax.set_ylim(limits[1])
        fig_ax.set_zlim(limits[2])

    ss = {
        # 'facecolor': 'deepskyblue',
        'linewidths': 0,
        'alpha': 0.75,
        'edgecolor': 'black',
        'antialiaseds': True,
    }

    # create colours for each segment
    magnitudes_scaled = copy.copy(magnitudes)
    if magnitudes_scaled is not None:
        min_magnitude = float(min(magnitudes_scaled))
        magnitudes_scaled = [v - min_magnitude for v in magnitudes_scaled]
        magnitudes_scaled = [v / float(max(magnitudes_scaled)) for v in magnitudes_scaled]  # scale 'magnitudes_scaled' within range [0, 1]

        cmap = cm.get_cmap("Spectral")  # Spectral
        magnitudes_scaled = cmap(magnitudes_scaled)

    norm = colors.Normalize(vmin=min(magnitudes),vmax=max(magnitudes))
    cmap = cm.get_cmap("Spectral")
    ss["facecolors"] = cmap(norm(magnitudes))

    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array([])
    fig.colorbar(m)

    collection = Poly3DCollection(verts, **ss)
    fig_ax.add_collection3d(collection)

    fig.savefig(file_name)

    return fig, fig_ax


def find_area_poly_3d(poly):
    #unit normal vector of plane defined by points a, b, and c
    def unit_normal(a, b, c):
        x = np.linalg.det([[1, a[1], a[2]],
                           [1, b[1], b[2]],
                           [1, c[1], c[2]]])
        y = np.linalg.det([[a[0], 1, a[2]],
                           [b[0], 1, b[2]],
                           [c[0], 1, c[2]]])
        z = np.linalg.det([[a[0], a[1], 1],
                           [b[0], b[1], 1],
                           [c[0], c[1], 1]])
        magnitude = (x**2 + y**2 + z**2)**.5
        return x/magnitude, y/magnitude, z/magnitude

    if len(poly) < 3:  # not a plane - no area
        return 0

    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)

