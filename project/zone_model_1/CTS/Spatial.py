import math
import numpy as np
from numpy import array
import transformations as mat


def poly_populate_vertices_along_segments(vertices, segments, inter_vertices_distance):
    if inter_vertices_distance == 0:
        return vertices, segments

    vertices_original = np.array(vertices, dtype=float).tolist()
    segments_original = np.array(segments, dtype=int).tolist()

    # Populate vertices by adding more vertices along segment edges
    for i, segment in enumerate(segments_original):
        # calculate useful variables
        vertex1_index, vertex2_index = segment
        vertex1 = vertices_original[vertex1_index]
        vertex2 = vertices_original[vertex2_index]
        length_segment = np.sqrt(np.sum(np.power(np.array(vertex1) - np.array(vertex2), 2)))
        total_inter_vertices = int(length_segment / inter_vertices_distance) - 1
        length_inter_segment = length_segment / float(total_inter_vertices + 1)

        # assign containers for current segment
        new_vertices_container = []
        new_segments_container = []
        total_vertices = len(vertices)

        # case for over 2 inter-vertices
        if total_inter_vertices >= 2:
            # log new vertices
            for index_node_in_small_segment in np.arange(total_inter_vertices):
                index_node_in_small_segment += 1
                length_from_n1 = index_node_in_small_segment * length_inter_segment
                new_vertex_in_small_segment = (np.array(vertex2) - np.array(vertex1)) * (
                length_from_n1 / length_segment)
                new_vertex_in_small_segment = np.array(vertex1) + new_vertex_in_small_segment
                new_vertex_in_small_segment = new_vertex_in_small_segment.tolist()
                new_vertices_container.append(new_vertex_in_small_segment)
            # log new segments
            for each_segment in np.arange(total_inter_vertices):  # [0, 1, 2, ... ]
                index_node_before = total_vertices + each_segment
                index_node_after = index_node_before + 1
                new_segments_container.append([index_node_before, index_node_after])
            # new_segments_container.append([total_vertices + index_node_in_small_segment, vertex2_index])
            # finalise vertices and segments log
            vertices.extend(new_vertices_container)
            segments.extend(new_segments_container)
            segments[i] = [vertex1_index, total_vertices]
            segments[-1] = [index_node_before, vertex2_index]

    return vertices, segments


def cartesian_to_polar(x,y,z):
    r = np.sqrt(np.power(x,2) + np.power(y,2) + np.power(z,2))
    theta = math.atan2(y,x)
    phi = math.atan2(np.sqrt(np.power(x,2) + np.power(y,2)), z)
    return r, theta, phi


def polar_to_cartesian(r,theta,phi):
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    return x, y, z


def rotate_3d_vertices(vertices, rotate_about_x, rotate_about_y, rotate_about_z):
    vertices_mat = np.matrix(vertices)
    row_size = np.size(vertices_mat,axis=0)
    zeros_column_mat = np.ones((row_size, 1), dtype=float)
    vertices_mat = np.append(vertices_mat,zeros_column_mat,axis=1)

    rx = np.matrix(mat.rotation_matrix(rotate_about_x,[1,0,0]))
    ry = np.matrix(mat.rotation_matrix(rotate_about_y,[0,1,0]))
    rz = np.matrix(mat.rotation_matrix(rotate_about_z,[0,0,1]))

    r = rx * ry * rz

    vertices_mat = vertices_mat.transpose()
    new_vertices = r * vertices_mat
    new_vertices = new_vertices[0:(np.size(new_vertices,0)-1),:]
    new_vertices = new_vertices.transpose()
    new_vertices = new_vertices.tolist()

    return new_vertices


def rotate_3d_vertices2(vertices, theta, axis):
    vertices_mat = np.matrix(vertices)
    row_size = np.size(vertices_mat,axis=0)
    zeros_column_mat = np.ones((row_size, 1), dtype=float)
    vertices_mat = np.append(vertices_mat,zeros_column_mat,axis=1)

    r = np.matrix(mat.rotation_matrix(theta, axis))

    vertices_mat = vertices_mat.transpose()
    new_vertices = r * vertices_mat
    new_vertices = new_vertices[0:(np.size(new_vertices,0)-1),:]
    new_vertices = new_vertices.transpose()
    new_vertices = new_vertices.tolist()

    return new_vertices


def get_plane_function(vertices, orientation=[0,0,0]):
    """
    DESCRIPTION:
        Calculate the plane which all vertex in vertices are on it. Plane is represented in a, b, c and d, which
        a x + b y + c z + d == 0
    PARAMETERS:
        vertices    {list}      a list with every individual cell describing a coordinate in (x, y, z)
    """
    # compute plane equation
    vertices_ = np.matrix(vertices)
    vertices_ = vertices_
    point1 = vertices_[0, :]
    point2 = vertices_[1, :]
    point3 = vertices_[2, :]

    vector1 = point2 - point1
    vector2 = point3 - point1

    norm = np.cross(vector1, vector2)
    norm = norm[0,:]

    d = -np.sum(np.array(norm)*np.array(point1))

    # check with given orientation
    is_the_same_sign = []
    for i, v in enumerate(orientation):
        if v == 0 or norm[i] == 0:
            is_the_same_sign.append(None)
        else:
            is_the_same_sign.append(norm[i] / norm[i] == v / abs(v))
    is_the_same_sign = bool(sum([v for v in is_the_same_sign if v is not None]))  # delete "None" values in the list
    if not is_the_same_sign:  # invert equation if required
        norm = [-v for v in norm]
        d = -d

    return norm[0], norm[1], norm[2], d


def angle_between_2vectors_3d(vector1, vector2):
    a = np.dot(vector1, vector2)
    b = np.linalg.norm(np.cross(vector1, vector2))
    theta = np.arctan2(b, a)
    return theta
