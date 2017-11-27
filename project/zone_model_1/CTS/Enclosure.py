# -*- coding: utf-8 -*-
import numpy as np
import Spatial
import triangle


class Surface(object):
    def __init__(self, vertices, connections, orientation=[0,0,0], if_proeceed=True, inter_vertices_distance=None,
                 triangulation_parameters="pq30a0.1D"):
        """
            PARAMETERS:
                vertices
                connections
        """
        self.vertices = vertices
        self.connections = connections
        self.givenOrientation = orientation

        self.norm = None  # normal of the plane (which accommodates 'vertices')
        self.axis = None  # axis which coplanar to the plane (which accommodates 'vertices') and horizontal to xy plane
        self.angle = None  # angle for the plane to rotate to be parallel to xy plane
        self.zValue = None  # z coordinates should be the same after rotation for all vertices
        self.vertices2D = None

        self.populateInterVerticesDistance = None
        # self.verticesPopulated = None
        # self.connectionsPopulated = None
        self.triangleMinAngle = None
        self.triangleMaxArea = None
        self.triangleVertices = None
        self.triangleConnections = None

        self.elements = None  # [coordinates, norm, centroid,]

        if if_proeceed:
            self.proceed_to_fragmentation(inter_vertices_distance,triangulation_parameters)

    def proceed_to_fragmentation(self, inter_vertices_distance=None, triangulation_parameters="pq30a0.1D"):
        self.transform_to_2d()
        # self.triangulation(inter_vertices_distance, min_allowable_angle, max_allowable_area)
        self.fragmentation(inter_vertices_distance, triangulation_parameters)
        self.transform_to_3d()
        self.elementalisation()

    def transform_to_2d(self):
        vertices = self.vertices
        # calculate the plane equation which all vertices are on
        a, b, c, d = Spatial.get_plane_function(vertices, self.givenOrientation)
        norm = np.array([a, b, c]) / np.linalg.norm([a,b,c])
        self.norm = norm

        if abs(norm[2]) != 1:
            # calculate the axis the plane is going to rotate about
            axis = np.cross(norm, [0, 0, 1])
            axis = np.array(axis) / np.linalg.norm(axis)
            self.axis = axis

            # calculate the angle the plane is going to rotate about the axis
            theta = Spatial.angle_between_2vectors_3d(norm, [0, 0, 1])
            self.angle = theta

            # calculate the matrix after rotation
            vertices1 = Spatial.rotate_3d_vertices2(vertices, theta, axis)
            a1, b1, c1, d1 = Spatial.get_plane_function(vertices1)
            norm1 = np.array([a1, b1, c1]) / np.linalg.norm([a1, b1, c1])
        else:
            axis = [1,0,0]
            theta = 0
            vertices1 = vertices
            norm1 = norm
            self.axis = axis
            self.angle = theta

        # log z-axis constant value, for later use
        z_value = None
        if abs(norm1[2]) == 1:
            z_value = vertices1[0]
            z_value = z_value[2]
        else:
            z_values = None
            print "!!!WARNING!!! NOT PARALLEL TO XY PLANE."
        self.zValue = z_value

        # log new vertices in 2D
        vertices1 = np.array(vertices1)
        vertices1 = vertices1[:, 0:2]
        self.vertices2D = vertices1.tolist()


    def fragmentation(self, inter_vertices_distance=None,triangulation_parameters="pq30a0.1D"):
        # instantiate 'vertices' and 'connections'
        vertices = np.array(self.vertices2D).tolist()
        connections = np.array(self.connections).tolist()

        # Triangulation the 'vertices grid'
        shape_poly = {"vertices": np.array(vertices, dtype=float),
                      "segments": np.array(connections, dtype=int)}
        cdt = triangle.triangulate(shape_poly,triangulation_parameters)
        self.triangleVertices = cdt["vertices"]
        self.triangleConnections = cdt["triangles"]


    def elementalisation(self):
        self.elements = []
        for each_triangle in self.triangleConnections:
            coors = [
                self.triangleVertices[each_triangle[0]],
                self.triangleVertices[each_triangle[1]],
                self.triangleVertices[each_triangle[2]]
            ]
            centroid = [
                (coors[0][0] + coors[1][0] + coors[2][0]) / 3.,
                (coors[0][1] + coors[1][1] + coors[2][1]) / 3.,
                (coors[0][2] + coors[1][2] + coors[2][2]) / 3.
            ]
            self.elements.append([coors, self.norm, centroid])


    def transform_to_3d(self):
        vertices2d = np.array(self.triangleVertices)

        # transform to 3d which is still parallel to xy plane
        n_rows = np.size(vertices2d, axis=0)
        ones_column = np.ones(n_rows) * self.zValue
        vertices3d = np.c_[vertices2d, ones_column]

        # transform to 3d which is back to its original state
        theta = -self.angle
        axis = self.axis
        vertices3d = Spatial.rotate_3d_vertices2(vertices3d, theta, axis)
        self.triangleVertices = vertices3d


class Assembly(object):
    def __init__(self, Surfaces):
        """
        DESCRIPTION:
            Collection of Surface objects, which can be thought as the enclosure.
        PROPERTIES:
            elements                            {list, -}               Each element of the list contains -
                                                                        [coordinates, norm, centroid,]
            elementsMatrixNakedViewFactors      {ndarray float, -}
        """
        self.surfaces = Surfaces

        self.elements = None
        self.originalLocation = None
        self.currentLocation = None
        self.elementsMatrixNakedViewFactors = None

        self.surface_matrix_builder()

    def surface_matrix_builder(self):
        self.elements = []
        self.originalLocation = []

        # Assign all elements to a master collection "self.elements". Their original locations are recorded at the same
        # time, each_surface.originalLocation = [surface_to_belong, element_index_in_that_surface]
        for i_surface, each_surface in enumerate(self.surfaces):
            for i_element, each_element in enumerate(each_surface.elements):
                self.elements.append(each_element)
                self.originalLocation.append([i_surface, i_element])

        # Create a master matrix M, which rows and columns are both elements from "self.elements". This matrix contains
        # view factors without dA, only cos(theta_1) * cos(theta_2) / pi / s^2.
        # Note that theta_1 and theta_2, the numberings 1 and 2 are denoted as i and j below
        self.elementsMatrix = np.zeros(shape=(len(self.elements),len(self.elements)))
        column_to_start_with = 0
        for i in np.arange(0,len(self.elements),1,dtype=int):
            column_to_start_with += 1
            for j in np.arange(column_to_start_with, len(self.elements), 1, dtype=int):
                vertex1_coors = self.elements[i][0]
                vertex1_norm = self.elements[i][1]
                vertex1_centroid = np.asarray(self.elements[i][2])
                vertex2_coors = self.elements[j][0]
                vertex2_norm = self.elements[j][1]
                vertex2_centroid = np.asarray(self.elements[j][2])

                vector12 = vertex2_centroid - vertex1_centroid
                vector21 = vertex1_centroid - vertex2_centroid

                # check if A_i and A_j are facing each other
                theta_1 = Spatial.angle_between_2vectors_3d(vertex1_norm, vector12)
                theta_2 = Spatial.angle_between_2vectors_3d(vertex2_norm, vector21)
                is_facing_each_other = (theta_1+theta_2 < 0.5*np.pi)

                # check if clear path between A_i to A_j
                is_obstacle_free = True

                # view factor calculation
                view_factor_ = 0
                if is_facing_each_other and is_obstacle_free:
                    s2 = np.sum(np.power(vector12,2))
                    view_factor_ = np.cos(theta_1) * np.cos(theta_2) / np.pi / s2
                self.elementsMatrixNakedViewFactors[i,j] = view_factor_

