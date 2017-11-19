"""
todo: doc string
"""
# -*- coding: utf-8 -*-

import copy

import numpy as np
import triangle

from project.func import tools
import warnings

# EXPERIMENTAL GEOMETRY SET 0:
# vertices sequence of geometry 1 (emitter):    [[5,0,0],   [5,0,5],    [10,0,5],   [10,0,0]]
# vertices sequence of geometry 2 (receiver):   [[5,10,0],  [15,10,0],  [15,10,10], [5,10,10]]

# todo: finish this function
def _poly_mesh_triangles(vertices, segments, internal_angle_max=30, area_max=0.2):
    # NOTE: Only works with vertices on the same plane
    normal_2d = (0,0,1)
    dimension_vertices = np.shape(vertices)[1]

    # Required inputs, making sure mutables are not modified
    vertices = copy.copy(vertices)
    segments = copy.copy(segments)

    # CHECK DIMENSIONS AND CONVERT TO 2D IF NECCESSARY
    if dimension_vertices==3:
        vertices_3d     = vertices
        normal_3d = tools.find_plane_norm(vertices_3d)
        vertices_3d_rotated, angle, axis, z_consistant = _poly_rotate_3d(vertices, normal_3d, normal_2d)
        vertices_2d     = vertices_3d_rotated[:,0:2]
    elif dimension_vertices==2:
        vertices_2d = vertices
    else:
        warnings.warn("Check vertices coordinates, only 2d and 3d are accepted.")

    # TRIANGULATION
    # Process triangulation parameters
    triangulation_parameters = "".join(["pq", str(internal_angle_max), "a", str(area_max), "D"])
    # Packing into dictionary
    poly = {
        "vertices": np.array(vertices_2d, dtype=float),
        "segments": np.array(segments, dtype=float)
    }
    # Triangulation
    poly_triangles = triangle.triangulate(poly, triangulation_parameters)

    # POSTPROCESSING - PREPARE FOR OUTPUT
    # Convert to original orientation and dimension
    if dimension_vertices==3:
        vertices_3d = np.zeros()
        vertices_3d[:,0:2] = poly_triangles["vertices"]
        vertices_3d[:,2] = z_consistant
        vertices_3d = __poly_rotate_3d(vertices_3d, normal_2d, normal_3d)
        vertices = vertices_3d
    segments = poly_triangles["segments"]

    return vertices, segments


def _poly_rotate_3d(vertices_3d, normal, normal_target=(0, 0, 1)):
    # todo: check requried variables (attributes) and their type
    if tuple(normal) == tuple(normal_target):
        z_consistant = np.average(vertices_transformed[:, 2].flatten())
        angle = np.nan
        axis = np.nan
        vertices_3d_transformed = vertices_3d
    else:
        # Calculate the rotation angle to achieve perpendicular to (0,0,1)
        angle   = tools.find_angle_between_vectors_3d(normal, normal_target)

        # Calculate the axis (vector) that rotate about to achieve perpendicular to (0,0,1)
        axis    = np.cross(normal, (0, 0, 1))
        axis    = axis / np.linalg.norm(axis)

        # Calculate the vertices which form a plane that perpendicular to (0,0,1)
        vertices_3d_transformed = tools.find_rotated_vertices_3d(vertices_3d, axis, angle)

        # Calculate the z value
        z_consistant = np.average(vertices_3d_transformed[:,2].flatten())


    return vertices_3d_transformed, angle, axis, z_consistant


# TODO: construct a class 'Surface' to accommodate individual plane
# TODO: construct a class 'Scene' to accommodate collection of plane
class Poly3D(object):
    def __init__(self, vertices, segments=None, normal=np.nan, temperature=np.nan, status=""):
        # INPUTS VARIABLE
        self.vertices               = np.array(vertices)
        self.segments               = segments
        self.normal                 = tools.find_plane_norm(self.vertices)
        self.temperature            = temperature
        self.status                 = status

        # INPUTS VARIABLE (INDIRECT)
        self.x                      = self.vertices[0,:]
        self.y                      = self.vertices[1,:]
        self.z                      = self.vertices[2,:]
        if segments is None:
            # if 'segments' is None data type, by default, the program will assume the input 'vertices' is in clockwise
            # order and self-enclosed. i.e. for vertices=[[1,1], [2,2], [3,3]], segments=[[0,1],[1,2],[2,0]]
            segments = [[i, i+1] for i, v in enumerate(np.zeros((len(vertices), 2)))]
            segments[-1][-1] = 0
            segments = np.asarray(segments)
            self.segments = segments

        # DERIVED VARIABLES BY "find_2d"
        self.vertices_2d            = []
        self.transformation_angle   = np.nan
        self.transformation_axis    = []
        self.transformation_z       = np.nan
        self.x_2d                   = []
        self.y_2d                   = []

    def find_normal(self):
        self.normal = tools.find_plane_norm(self.vertices)

    def get_normal(self):
        return self.normal

    def get_vertices(self):
        return self.vertices

    def get_vertices_individual(self):
        return self.x, self.y, self.z

    def get_segments(self):
        return self.segments



class Scene3D(object):
    def __init__(self):
        self.surfaces = []

        self.vertices = []
        self.normals = []
        self.temperatures = []
        self.status = []

        self.x = []
        self.y = []
        self.z = []

    def add_surface(self, surface=Poly3D, skip_assignment_to_collection=False):
        # Check if the input 'surface' is the Surface object data type
        if surface is not Surface3D:
            warnings.warn("Input 'surface' is not Surface object data type.")
        else:
            self.surfaces.append(surface)

        # This check is in place for assignment of multiple surfaces and their one or all properties are the same and
        # known, which they can be assigned manually to save computing time.
        if skip_assignment_to_collection:
            return 0

        # Append surface properties to collection
        self.vertices.append(       surface.vertices)
        self.normals.append(        surface.normal)
        self.status.append(         surface.status)
        self.temperatures.append(   surface.temperature)
        self.x.append(              surface.x)
        self.y.append(              surface.y)
        self.z.append(              surface.z)

        return 0

    def get_limits_x(self):
        return np.min(self.x), np.max(self.x)

    def get_limits_y(self):
        return np.min(self.y), np.max(self.y)

    def get_limits_z(self):
        return np.min(self.z), np.max(self.z)


class ViewFinder3(object):
    def __init__(self, scene=Scene3D):
        pass


class EnclosureEnvironment(object):
    """
    _raw_vertices_collection            A list of all input vertices coordinates
    _raw_vertex_sequences_collection    A list of vertex index sequences represents raw planes
    _planes_vertices                    A list of all planes' vertices coordinates
    _planes_temperature                 A list of all planes' temperature
    _planes_status                      A list of all planes' status: emitter, receiver, inert
    _planes_norm                        A list of all planes' norm
    _debris_vertices_collection         An array of all debris' vertices coordinates
    _debris_vertex_sequences            An array of vertex index sequences represents debris
    _debris_vertices                    An array of all debris' vertices
    _debris_plane_index                 An array of integers indicating which
    _debris_temperature                 An array of all debris' temperature
    _debris_status                      An array of all debris' status
    _debris_norm                        An array of all debris' normal vector
    """
    def __init__(self):
        self._planes_vertices_collection = []
        self._planes_vertex_sequences_collection = []
        self._planes_vertices = []
        self._planes_temperature = []
        self._planes_status = []
        self._planes_norm = []
        self._debris_vertices_collection = None  # ndarray
        self._debris_vertex_sequences_collection = None  # ndarray
        self._debris_vertices = None  # ndarray
        self._debris_plane_index = None  # ndarray
        self._debris_status = None  # ndarray
        self.__receivers_index = None  # ndarray
        self.__emitters_index = None  # ndarray
        self._debris_radiant_heat_flux = None

        self.__debris_total_receivers = np.nan
        self.__debris_total_emitters = np.nan
        self.__debris_total_inerts = np.nan

    # STEP 0: Define planes vertices
    def define_planes(self, list_planes_vertices, list_planes_temperature, list_planes_status):
        # translate plane status from string to number - emitter: 0, receiver: 1, inert: 2
        status_map = {"emitter": 0, "receiver": 1, "inert": 2}
        list_planes_status = [status_map[v] for v in list_planes_status]

        self._planes_vertices = list_planes_vertices
        self._planes_temperature = list_planes_temperature
        self._planes_status = list_planes_status

    # STEP 1: Make plane (re-arrange data structure)
    def make_plane(self):
        # make a collection of all vertices, sequences & planes' norm
        for l in self._planes_vertices:
            # sequence
            len1 = len(self._planes_vertices_collection)
            len2 = len1 + len(l)
            vertex_sequences = zip(range(len1, len2), range(len1 + 1, len2 + 1))
            vertex_sequences = [list(vertex) for vertex in list(vertex_sequences)]
            vertex_sequences[-1][-1] = vertex_sequences[0][0]
            self._planes_vertex_sequences_collection.append(vertex_sequences)

            # vertices
            for ll in l:
                self._planes_vertices_collection.append(ll)

            # norm
            self._planes_norm.append(tools.find_plane_norm(l))

    # STEP 2: Make debris (triangulation)
    def make_debris(self, triangulation_parameters = "pq30a0.2D"):
        """
        :return self._debris_vertex_sequence:
        :return self._debris_vertices_collection:
        :return self._debris_plane_index:
        """

        for i, plane_vertices in enumerate(self._planes_vertices):
            plane_norm = self._planes_norm[i]
            target_norm = (0, 0, 1)

            # calculate the axis the plane is going to rotate about
            rotation_axis = np.cross(plane_norm, target_norm)
            rotation_axis = np.array(rotation_axis) / np.linalg.norm(rotation_axis)

            # calculate the angle the plane is going to rotate about the axis
            rotation_angles = tools.find_angle_between_vectors_3d(plane_norm, [0, 0, 1])

            # calculate the rotated vertices (array)
            plane_vertices_rotated = tools.find_rotated_vertices_3d(plane_vertices, rotation_axis, rotation_angles)

            # calculate Triangulation
            plane_connections = [[i, i+1] for i,v in enumerate(np.zeros((len(plane_vertices_rotated), 2)))]
            plane_connections[-1][-1] = 0

            shape_poly = {"vertices": np.array(plane_vertices_rotated[:, 0:2], dtype=float),
                          "segments": np.array(plane_connections, dtype=int)}
            cdt = triangle.triangulate(shape_poly, triangulation_parameters)

            # calculate reversely rotated triangulated vertices (restore to original orientation with debris)
            debris_vertices = cdt["vertices"]
            debris_vertices = np.append(debris_vertices,
                                        np.ones((len(debris_vertices), 1)) * np.average(plane_vertices_rotated[:,2]),
                                        axis=1)
            debris_vertices = tools.find_rotated_vertices_3d(debris_vertices, rotation_axis, -rotation_angles)

            # assign debris vertex sequences collection, vertices collection and debris' plane index to the object
            if self._debris_vertex_sequences_collection is None:
                self._debris_vertex_sequences_collection = copy.copy(cdt["triangles"])
                self._debris_vertices_collection = copy.copy(debris_vertices)
                self._debris_plane_index = np.ones((np.shape(cdt["triangles"])[0], 1), dtype=int) * i
                self._debris_status = np.ones((np.shape(cdt["triangles"])[0], 1), dtype=int) * self._planes_status[i]
            else:
                self._debris_vertex_sequences_collection = np.append(
                    self._debris_vertex_sequences_collection,
                    cdt["triangles"] + len(self._debris_vertices_collection),
                    axis=0
                )
                self._debris_vertices_collection = np.append(
                    self._debris_vertices_collection,
                    debris_vertices,
                    axis=0
                )
                self._debris_plane_index = np.append(
                    self._debris_plane_index,
                    np.ones((len(cdt["triangles"]), 1), dtype=int) * i,
                    axis=0
                )
                self._debris_status = np.append(
                    self._debris_status,
                    np.ones((len(cdt["triangles"]), 1), dtype=int) * self._planes_status[i],
                    axis=0
                )

        # assign each debris' vertices.
        shape_old = np.shape(self._debris_vertex_sequences_collection)
        debris_vertex_sequences_flatten = self._debris_vertex_sequences_collection.flatten(order="C")
        debris_vertices = self._debris_vertices_collection[debris_vertex_sequences_flatten]
        debris_vertices = np.reshape(debris_vertices, (shape_old[0], shape_old[1], 3), order="C")
        self._debris_vertices = debris_vertices

        # find each debris' status
        self.__debris_total_emitters = np.count_nonzero(self._debris_status == 0)
        self.__debris_total_receivers = np.count_nonzero(self._debris_status == 1)
        self.__debris_total_inerts = np.count_nonzero(self._debris_status == 2)

        # CHECK NUMBERS
        print("count self.__debris_vertices")

    def _make_matrix_for_radiation_calculation(self):
        a_e = np.ones(shape=(self.__debris_total_emitters,1), dtype=float)
        a_r = np.ones(shape=(self.__debris_total_receivers,1), dtype=float)

        # STEP 1: CONSTRUCT MATRIX CONTAINER
        radiation_mat = np.zeros(shape=(self.__debris_total_receivers, self.__debris_total_emitters), dtype=float)
        # # mat_A_r         = A_r * np.transpose(a_e)
        # # mat_A_e         = a_r * np.transpose(A_e)
        # mat_T_r         = T_r * np.transpose(a_e)
        # mat_T_e         = a_r * np.transpose(T_e)
        # # mat_theta_r
        # # mat_theta_e
        # # mat_r2          =
        # mat_phi
        # mat_E
        emitters_mat = np.zeros(shape=(self.__debris_total_emitters, 1), dtype=float)
        receivers_mat = np.zeros(shape=(self.__debris_total_receivers, 1), dtype=float)

        # STEP 2: ITERATE THROUGH ALL PATHS AND CALCULATE THE RADIATION
        # Obtain receivers' and emitters' index (of the debris collection) for later use.
        index_debris = np.arange(0, len(self._debris_plane_index))
        radiation_mat_receivers_index = index_debris[self._debris_status.flatten()==1]
        radiation_mat_emitters_index = index_debris[self._debris_status.flatten()==0]

        # start loop through all emitters for every receivers.
        for r in np.arange(0, np.shape(radiation_mat)[0]):
            r_debris_index = radiation_mat_receivers_index[r]
            r_plane_index = self._debris_plane_index[r_debris_index, 0]
            r_vertices = self._debris_vertices[r_debris_index]
            r_area = tools.find_area_poly_3d(r_vertices)
            r_norm = self._planes_norm[r_plane_index]
            r_centroid = np.average(r_vertices, axis=0)
            r_temperature = self._planes_temperature[r_plane_index]
            for e in np.arange(0, np.shape(radiation_mat)[1]):
                e_debris_index = radiation_mat_emitters_index[e]
                e_plane_index = self._debris_plane_index[e_debris_index, 0]
                e_vertices = self._debris_vertices[e_debris_index]
                e_area = tools.find_area_poly_3d(e_vertices)
                e_norm = self._planes_norm[e_plane_index]
                e_centroid = np.average(e_vertices, axis=0)
                e_temperature = self._planes_temperature[e_plane_index]

                vector_1_2 = r_centroid - e_centroid
                vector_2_1 = e_centroid - r_centroid

                theta_1 = tools.find_angle_between_vectors_3d(e_norm, vector_2_1)  # todo: currently incorrect
                theta_2 = tools.find_angle_between_vectors_3d(r_norm, vector_1_2)  # todo: currently incorrect

                distance = np.sqrt(np.sum(np.power(r_centroid-e_centroid, 2)))

                configuration_factor = np.cos(theta_1) * np.cos(theta_2) / np.pi / np.power(distance, 2) * e_area

                emissivity = 0.9
                sigma = 5.67e-8
                radiation_emission = emissivity * sigma * (e_temperature**4 - r_temperature**4)

                radiation_mat[r, e] = configuration_factor * radiation_emission


        # STEP 3: ASSIGN RESULTS BACK TO INDEXED FORM
        debris_radiant_heat_flux_r = np.sum(radiation_mat, axis=1).flatten()
        self._debris_radiant_heat_flux = self._debris_plane_index * 0.
        for i, v in enumerate(radiation_mat_receivers_index):
            self._debris_radiant_heat_flux[v] = debris_radiant_heat_flux_r[i]

        print("break")


    def _dev_test(self, sample=0):
        # INPUTS
        vertices = [
            [[5, 0, 0],     [5, 0, 5],      [10, 0, 5],     [10, 0, 0]],
            [[5, 5, 0],    [15, 5, 0],    [15, 5, 10],   [5, 5, 10]]
        ]
        temperatures = [5000, 275]
        status = ['emitter', 'receiver']

        # PLANE CALCULATION
        self.define_planes(vertices, temperatures, status)

        self.make_plane()
        print(
            "PLANES VERTICES: \n{}\n"
            "PLANES VERTEX SEQUENCE: \n{}\n"
            "PLANES NORMAL: \n{}\n"
            .format(
                self._planes_vertices_collection,
                self._planes_vertex_sequences_collection,
                self._planes_norm
            )
        )

        # DEBRIS CALCULATION
        self.make_debris()
        print(
            "DEBRIS VERTEX COLLECTION:\n{0}\n"
            "DEBRIS PLANE INDEX:\n{1}\n"
            "DEBRIS SHAPE VERTICES:\n{2}\n"
            "TOTAL EMITTERS:\n{3}\n"
            "TOTAL RECEIVERS:\n{4}\n"
            "TOTAL INERTS:\n{5}\n"
            .format(
                self._debris_vertices_collection,
                self._debris_plane_index,
                self._debris_vertices,
                self.__debris_total_emitters,
                self.__debris_total_receivers,
                self.__debris_total_inerts
            )
        )

        self._make_matrix_for_radiation_calculation()

        print("maximum heat flux: ", max(self._debris_radiant_heat_flux))

        limites = ((0,15),(0,15),(0,15))

        fig, fig_ax = plot_poly_axes3d('test.png', self._debris_vertices, limits=limites, magnitudes=list(self._debris_radiant_heat_flux.flatten()))
        fig.show()

        print("test break")

    def make_inputs_from_text(self, input_text):
        inputs_parsed = tools.parse_inputs_from_text(input_text)

        inputs_parsed["vertices"]
        inputs_parsed["temperatures"]
        inputs_parsed["status"]

        print(inputs_parsed)


if __name__ == "__main__":
    e = EnclosureEnvironment()
    e._dev_test()