# -*- coding: utf-8 -*-
from numpy import array
import numpy as np
import math
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl
import time
from inspect import currentframe, getframeinfo  # for error handling, get current line number
import csv
import os

from matplotlib import rcParams
# Say, "the default sans-serif font is COMIC SANS"
# matplotlib.rcParams['font.serif'] = "Helvetica"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = ["serif"]



# DESCRIPTION:
# This function lowers the data_arr resolution by res_int
# PARAMETERS:
# data_arr      input data which to be processed. 1d ndarray which has np.shape = (n,), or a list of 1d ndarray.
# res_int       sample resolution. e.g. res_int = 60 means sample one value (the first one) from 60 original values.
def adjust_resolution_1d(data_arr, res_int):
    def helper_func(data_arr_):
        data_arr_ = np.reshape(data_arr_, (len(data_arr_),))
        rows_total = len(data_arr)
        cols = int(res_int)
        rows = int(rows_total/cols) + 1
        filter_arr = np.zeros(shape=(rows, cols))
        filter_arr[:, 0] = 1
        filter_arr = np.reshape(filter_arr, (rows*cols,))
        filter_arr = filter_arr[0:rows_total]
        data_arr_new = np.ma.masked_where(filter_arr<1, data_arr).compressed()
        return data_arr_new

    # take account of a list of ndarray
    data_arr = [helper_func(v) for v in data_arr] if len(np.shape(data_arr)) > 1 else helper_func(data_arr)

    return data_arr


def read_text_to_list(file_full_path):
    f = open(file_full_path)

    csv_f = csv.reader(f)

    data_list = [v for v in csv_f]

    f.close()
    return data_list


def write_list_to_csv(file_name, a_list):
    with open(file_name,'wb') as f:
        output_writer = csv.writer(f, dialect='excel')
        output_writer.writerows(a_list)


class PlotScatter2D(object):
    def __init__(self):
        self.figure = plt.figure()
        self.axes = []
        self.lines = []
        self.texts = []

    def self_delete(self):
        plt.close(self.figure)
        self.figure = None
        self.axes = None
        self.lines = None
        self.texts = None
        del self

    def plot(self, xyl1, xyl2=None):
        # create axes
        self.axes.append(self.figure.add_subplot(111))
        self.axes.append(self.axes[0].twinx()) if xyl2 is not None else None

        # create lines
        for i in tuple(xyl1):
            x, y, l = tuple(i)
            line = self.axes[0].plot(x, y, label=l)
            self.lines.append(line[0])
        if xyl2 is not None:
            for i in xyl2:
                x, y, l = tuple(i)
                line = self.axes[1].plot(x, y, label=l)
                self.lines.append(line[0])

    def format(self, **kwargs):
        # set format
        self.format_figure(**kwargs)
        self.format_axes(**kwargs)
        self.format_lines(**kwargs)
        self.format_legend(**kwargs)

        self.figure.tight_layout()

    def format_figure(self, **kwargs):
        figure_size_width = 10. if 'figure_size_width' not in kwargs else kwargs['figure_size_width']
        figure_size_height = 7.5 if 'figure_size_height' not in kwargs else kwargs['figure_size_height']
        figure_size_scale = 1. if 'figure_size_scale' not in kwargs else kwargs['figure_size_scale']
        figure_title = '' if 'figure_title' not in kwargs else kwargs['figure_title']
        figure_title_font_size = 15. if 'figure_title_font_size' not in kwargs else kwargs['figure_title_font_size']

        if figure_size_scale * figure_size_height > 0 and figure_size_scale * figure_size_width > 0:
            self.figure.set_size_inches(w=figure_size_width * figure_size_scale, h=figure_size_height * figure_size_scale)
        self.figure.suptitle(figure_title, fontsize=figure_title_font_size)
        self.figure.set_facecolor((1 / 237., 1 / 237., 1 / 237., 0.0))

    def format_axes(self, **kwargs):
        has_secondary = len(self.axes) > 1
        axis_lim_y2 = None
        axis_label_y2 = None

        axis_label_x = '' if 'axis_label_x' not in kwargs else kwargs['axis_label_x']
        axis_label_y1 = '' if 'axis_label_y1' not in kwargs else kwargs['axis_label_y1']
        if has_secondary:
            axis_label_y2 = '' if 'axis_label_y2' not in kwargs else kwargs['axis_label_y2']
        axis_label_font_size = 11. if 'axis_label_font_size' not in kwargs else kwargs['axis_label_font_size']
        axis_tick_font_size = 9. if 'axis_tick_font_size' not in kwargs else kwargs['axis_tick_font_size']
        axis_lim_x = self.axes[0].get_xlim() if 'axis_lim_x' not in kwargs else kwargs['axis_lim_x']
        axis_lim_y1 = self.axes[0].get_ylim() if 'axis_lim_y1' not in kwargs else kwargs['axis_lim_y1']
        if has_secondary:
            axis_lim_y2 = self.axes[1].get_ylim() if 'axis_lim_y2' not in kwargs else kwargs['axis_lim_y2']
        axis_linewidth = 1. if 'axis_linewidth' not in kwargs else kwargs['axis_linewidth']
        axis_scientific_format_x = False if 'axis_scientific_format_x' not in kwargs else kwargs['axis_scientific_format_x']
        axis_scientific_format_y1 = False if 'axis_scientific_format_y1' not in kwargs else kwargs['axis_scientific_format_y1']
        axis_scientific_format_y2 = False if 'axis_scientific_format_y2' not in kwargs else kwargs['axis_scientific_format_y2']

        axis_tick_width = .5 if 'axis_tick_width' not in kwargs else kwargs['axis_tick_width']
        axis_tick_length = 2.5 if 'axis_tick_length' not in kwargs else kwargs['axis_tick_length']


        self.axes[0].set_xlim(axis_lim_x)
        self.axes[0].set_ylim(axis_lim_y1)
        self.axes[1].set_ylim(axis_lim_y2) if has_secondary else None
        self.axes[0].set_xlabel(axis_label_x, fontsize=axis_label_font_size)
        self.axes[0].set_ylabel(axis_label_y1, fontsize=axis_label_font_size)
        self.axes[1].set_ylabel(axis_label_y2, fontsize=axis_label_font_size) if has_secondary else None
        self.axes[0].get_xaxis().get_major_formatter().set_useOffset(axis_scientific_format_x)
        self.axes[0].get_yaxis().get_major_formatter().set_useOffset(axis_scientific_format_y1)
        self.axes[1].get_yaxis().get_major_formatter().set_useOffset(axis_scientific_format_y2) if has_secondary else None

        [i.set_linewidth(axis_linewidth) for i in self.axes[0].spines.itervalues()]
        [i.set_linewidth(axis_linewidth) for i in self.axes[1].spines.itervalues()] if has_secondary else None

        self.axes[0].tick_params(axis='both', which='major', labelsize=axis_tick_font_size, width=axis_tick_width, length=axis_tick_length, direction='in')
        self.axes[0].tick_params(axis='both', which='minor', labelsize=axis_tick_font_size, width=axis_tick_width, length=axis_tick_length, direction='in')
        self.axes[1].tick_params(axis='both', which='major', labelsize=axis_tick_font_size, width=axis_tick_width, length=axis_tick_length, direction='in') if has_secondary else None
        self.axes[1].tick_params(axis='both', which='minor', labelsize=axis_tick_font_size, width=axis_tick_width, length=axis_tick_length, direction='in') if has_secondary else None


    def format_lines(self, **kwargs):
        # -----
        marker_size = 2 if 'marker_size' not in kwargs else kwargs['marker_size']
        mark_every = 100 if 'mark_every' not in kwargs else kwargs['mark_every']
        marker_fill_style = 'none' if 'marker_fill_style' not in kwargs else kwargs['marker_fill_style']
        marker_edge_width = .5 if 'marker_edge_width' not in kwargs else kwargs['marker_edge_width']
        # -----
        line_width = 1. if 'line_width' not in kwargs else kwargs['line_width']
        line_style = '-' if 'line_style' not in kwargs else kwargs['line_style']

        c = [(80, 82, 199), (30, 206, 214), (179, 232, 35), (245, 198, 0), (255, 89, 87)]
        c = [(colour[0] / 255., colour[1] / 255., colour[2] / 255.) for colour in c] * 100

        m = ['o', '^', 's', 'v', 'p', '*', 'D', 'd', '8', '1', 'h', '+', 'H'] * 40

        for i, l in enumerate(self.lines):
            l.set_marker(m[i])
            l.set_color(c[i])
            l.set_markersize(marker_size)
            l.set_markevery(mark_every)
            l.set_markeredgecolor(c[i])
            l.set_markeredgewidth(marker_edge_width)
            l.set_fillstyle(marker_fill_style)
            l.set_linestyle(line_style)
            l.set_linewidth(line_width)

    def format_legend(self, **kwargs):
        legend_is_shown = True if 'legend_is_shown' not in kwargs else kwargs['legend_is_shown']
        legend_loc = 0 if 'legend_loc' not in kwargs else kwargs['legend_loc']
        legend_font_size = 8 if 'legend_font_size' not in kwargs else kwargs['legend_font_size']
        legend_alpha = 1.0 if 'legend_alpha' not in kwargs else kwargs['legend_alpha']
        legend_is_fancybox = False if 'legend_is_fancybox' not in kwargs else kwargs['legend_is_fancybox']
        legend_line_width = 1. if 'legend_line_width' not in kwargs else kwargs['legend_line_width']

        line_labels = [l.get_label() for l in self.lines]
        legend = self.axes[len(self.axes) - 1].legend(
            self.lines,
            line_labels,
            loc=legend_loc,
            fancybox=legend_is_fancybox,
            prop={'size': legend_font_size}
        )
        legend.set_visible(True) if legend_is_shown else legend.set_visible(False)
        legend.get_frame().set_alpha(legend_alpha)
        legend.get_frame().set_linewidth(legend_line_width)

        self.texts.append(legend)

    def add_lines(self, xyl, axis=0):
        for i in xyl:
            x, y, l = tuple(i)
            line = self.axes[axis].plot(x, y, label=l)
            self.lines.append(line[0])

    def update_legend(self, **kwargs):
        """
        refresh the legend to the existing recent plotted lines.
        """
        self.texts[0].remove()

        # legend_is_shown = True if 'legend_is_shown' not in kwargs else kwargs['legend_is_shown']
        # legend_loc = 0 if 'legend_loc' not in kwargs else kwargs['legend_loc']
        # legend_font_size = 8 if 'legend_font_size' not in kwargs else kwargs['legend_font_size']
        # legend_alpha = 1.0 if 'legend_alpha' not in kwargs else kwargs['legend_alpha']
        # legend_is_fancybox = False if 'legend_is_fancybox' not in kwargs else kwargs['legend_is_fancybox']
        #
        # line_labels = [l.get_label() for l in self.lines]
        # legend = self.axes[len(self.axes)-1].legend(
        #     self.lines,
        #     line_labels,
        #     loc=legend_loc,
        #     fancybox=legend_is_fancybox,
        #
        #     prop={'size': legend_font_size}
        # )
        # legend.set_visible(True) if legend_is_shown else legend.set_visible(False)
        # legend.get_frame().set_alpha(legend_alpha)
        #
        # self.texts[0] = legend

        self.format_legend(**kwargs)

    def update_format_line(self, line_name, **kwargs):
        lines_index = {}
        for i,v in enumerate(self.lines):
            lines_index.update({v.get_label(): i})
        i = lines_index[line_name] if line_name in lines_index else None

        if i is None:
            frame_info = getframeinfo(currentframe())
            print('ERROR: Line name does not exist.')
            print('File: ' + str(frame_info.filename))
            print('Line: ' + str(frame_info.lineno))
            return None

        line_style = self.lines[i].get_linestyle() if 'line_style' not in kwargs else kwargs['line_style']
        line_width = self.lines[i].get_linewidth() if 'line_width' not in kwargs else kwargs['line_width']
        color = self.lines[i].get_color() if 'color' not in kwargs else kwargs['color']
        marker = self.lines[i].get_marker() if 'marker' not in kwargs else kwargs['marker']
        marker_size = self.lines[i].get_markersize() if 'marker_size' not in kwargs else kwargs['marker_size']
        mark_every = self.lines[i].get_markevery() if 'mark_every' not in kwargs else kwargs['mark_every']
        marker_edge_color = self.lines[i].get_markeredgecolor() if 'marker_edge_color' not in kwargs else kwargs['marker_edge_color']
        marker_edge_width = self.lines[i].get_markeredgewidth() if 'marker_edge_width' not in kwargs else kwargs['marker_edge_width']
        marker_fill_style = self.lines[i].get_fillstyle() if 'marker_fill_style' not in kwargs else kwargs['marker_fill_style']

        self.lines[i].set_linestyle(line_style)
        self.lines[i].set_linewidth(line_width)
        self.lines[i].set_color(color)
        self.lines[i].set_marker(marker)
        self.lines[i].set_markersize(marker_size)
        self.lines[i].set_markevery(mark_every)
        self.lines[i].set_markeredgecolor(marker_edge_color)
        self.lines[i].set_markeredgewidth(marker_edge_width)
        self.lines[i].set_fillstyle(marker_fill_style)

    def update_line(self, line_name):
        pass

    def remove_line(self, line_name):
        pass

    def save_figure(self, figure_name, time_suffix=False):
        time_suffix = False
        if time_suffix:
            figure_name += (" " + time.strftime("%m%d.%H%M%S"))
        self.figure.tight_layout()
        figure_name += '.pdf'
        self.figure.savefig(figure_name, bbox_inches='tight')

if __name__ == "__main__":
    x = np.arange(0,2*np.pi,0.01)
    y = np.sin(x)
    p = PlotScatter2D()
    p.plot([[x, np.sin(x), 'testing legend 1'], [x, np.cos(x), 'testing legend 2']], [[x, np.tan(x+1), 'testing legend 3']])
    p.format(
        figure_title='TITLE TESTING',
        axis_label_x='x axis label testing',
        figure_size_scale=0.5,
        figure_name='testing_figure_name',
        axis_lim_x=[0,2.*np.pi],
        axis_lim_y2=[-2,2],
    )
    p.save_figure("hello")


def read_file_to_array(file_name):
    with open(file_name,'r') as f:
        l = [x.strip(',').split() for x in f.readlines()]

    vertices = []
    for k in l:
        x, y = [items for items in l[k]]
        vertices.append([float(x), float(y)])

    return l


def read_file_to_dict(file_name):
    with open(file_name, 'r') as f:
        l = [x.strip(',').split() for x in f.readlines()]


def read_poly(file_name):
    '''
    Simple poly-file reader, that creates a python
         dictionary
    with information about vertices, edges and holes.
    It assumes that vertices have no attributes or
         boundary markers.
    It assumes that edges have no boundary markers.
    No regional attributes or area constraints are
    '''
    output = {'vertices': None,
              'holes': None,
              'segments': None}

    # open file and store lines in a list
    file = open(file_name, 'r')
    lines = file.readlines()
    file.close()
    lines = [x.strip('\n').split() for x in lines]

    # Store vertices
    vertices = []
    N_vertices, dimension, attr, bdry_markers = [int(x) for x in lines[0]]

    # We assume attr = bdrt_markers = 0
    for k in range(N_vertices):
        label, x, y = [items for items in lines[k + 1]]
        vertices.append([float(x), float(y)])
    output['vertices'] = array(vertices)

    # Store segments
    segments = []
    N_segments, bdry_markers = [int(x) for x in lines[N_vertices + 1]]
    for k in range(N_segments):
        label, pointer_1, pointer_2 = [items for items in lines[N_vertices + k + 2]]
        segments.append([int(pointer_1) - 1, int(pointer_2) - 1])
    output['segments'] = array(segments)

    # Store holes
    N_holes = int(lines[N_segments + N_vertices + 2][0])
    holes = []
    for k in range(N_holes):
        label, x, y = [items for items in lines[N_segments + N_vertices + 3 + k]]
        holes.append([float(x), float(y)])
    output['holes'] = array(holes)
    return output


def plot_poly_2d(file_name, vertices, segments):
    fig = plt.figure()
    fig_ax = fig.add_subplot(111)
    # fig_ax.set_xlim([-10, 310])
    # fig_ax.set_ylim([-10, 310])
    for each_segment in segments:
        vertices_container = []
        for each_vertex in each_segment:
            vertices_container.append(vertices[each_vertex])
        for index_vertices in np.arange(len(vertices_container)-1):
            vertex1 = vertices_container[index_vertices]
            vertex2 = vertices_container[index_vertices+1]
            x = [vertex1[0], vertex2[0]]
            y = [vertex1[1], vertex2[1]]
            fig_ax.plot(x, y, color='red', marker='o', markeredgecolor='black', markerfacecolor='black', markersize=2)
        vertex1 = vertices_container[0]
        x = [vertex2[0], vertex1[0]]
        y = [vertex2[1], vertex1[1]]
        fig_ax.plot(x, y, color='red', marker='o', markeredgecolor='black', markerfacecolor='black', markersize=2)
    fig.savefig(file_name, format='png', dpi=300)
    return fig


def plot_poly_3d(file_name, vertices, segments,
                 title=None, limits=None, no_marker=False, no_line=False, figure=None, figure_ax=None):
    # instantiate figure
    if not figure_ax:
        fig = plt.figure()
        fig_ax = fig.add_subplot(111, projection='3d')
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

    for each_segment in segments:
        vertices_container = []
        for each_vertex in each_segment:
            vertices_container.append(vertices[each_vertex])
        for index_vertices in np.arange(len(vertices_container)-1):
            vertex1 = vertices_container[index_vertices]
            vertex2 = vertices_container[index_vertices+1]
            x = [vertex1[0], vertex2[0]]
            y = [vertex1[1], vertex2[1]]
            z = [vertex1[2], vertex2[2]]
            if not no_line:
                fig_ax.plot(x, y, z, color='deepskyblue', marker='o', markeredgecolor='b', markerfacecolor='b', markersize=2)
            if not no_marker:
                fig_ax.scatter(x, y, z, color='b', marker='o')
        vertex1 = vertices_container[0]
        x = [vertex2[0], vertex1[0]]
        y = [vertex2[1], vertex1[1]]
        z = [vertex2[2], vertex1[2]]
        if not no_line:
            fig_ax.plot(x, y, z, color='deepskyblue', marker='o', markeredgecolor='b', markerfacecolor='b', markersize=2)
    fig.savefig(file_name, format='png', dpi=300)
    return fig, fig_ax


def plot_poly_axes3d(file_name, poly_coordinates, magnitudes=None,
                     title=None, limits=None, figure=None, figure_ax=None, width_scale=1):
    '''
    PARAMETERS:
        file_name           {string, -}         File name with extension i.e. '.png'. Plot will be saved according to this
                                                name
        poly_coordinates    {list float, -}     It contains a list of polygon. i.e. [subpoly1, subpoly2, ...]
                                                Each 'subpoly' is defined by a sequence of coordinates.
                                                i.e. [[x1, y1, z1], [x2, y2, z2], ...]
        magnitudes          {list float, -}     A list of number
    REMARKS:
        1. if figure_ax is not None, then it has to be an Axes3D object.
    '''

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

    # convert the current segment from vertex index to vertex coordinates
    # coordinates = []
    # for each_segment in segments:
    #     coordinates.append([tuple(vertices[each_vertex_index]) for each_vertex_index in each_segment])

    plot_style_default = {
        'facecolor': 'deepskyblue',
        'linewidths': 1,
        'alpha': 0.5,
        'edgecolor': 'black',
        'antialiaseds': True
    }

    # create colours for each segment
    if magnitudes is not None:
        max_magnitude = float(max(magnitudes))
        min_magnitude = float(min(magnitudes))
        magnitudes = [v - min_magnitude for v in magnitudes]
        magnitudes = [v/float(max(magnitudes)) for v in magnitudes]  # scale 'magnitudes' to [0, 1]

        cmap = cm.get_cmap('Spectral')  # define colour map
        magnitudes = cmap(magnitudes)

        plot_style_default['facecolor'] = magnitudes

    collection = Poly3DCollection(poly_coordinates, **plot_style_default)
    fig_ax.add_collection3d(collection)

    fig.savefig(file_name, format='png', dpi=300)
    return fig, fig_ax


def list_all_files_with_suffix(directory_str, suffix_str):
    list_files = []

    for f in os.listdir(directory_str):
        if f.endswith(suffix_str):
            list_files.append(os.path.join(directory_str, f))

    return list_files