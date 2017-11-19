# -*- coding: utf-8 -*-
from numpy import array
import numpy as np
import math
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import time
from inspect import currentframe, getframeinfo  # for error handling, get current line number
import csv
import os
import inspect


class Scatter2D(object):
    def __init__(self):
        self.figure = plt.figure()
        self.axes = []
        self.lines = []
        self.texts = []
        self._format = None

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

    def plot2(self, x, y, label="", second_axis=False):
        self.axes.append(self.figure.add_subplot(111))

        if second_axis:
            self.axes.append(self.axes[0].twinx())
            line = self.axes[1].plot(x, y, label=label)
        else:
            line = self.axes[0].plot(x, y, label=label)

        self.lines.append(line[0])

    def format(self, **kwargs):
        def map_dictionary(list_, dict_master):
            dict_new = dict()
            for key in list_:
                if key in dict_master:
                    dict_new[key] = dict_master[key]
            return dict_new

        dict_inputs_figure = map_dictionary(inspect.signature(self.format_figure).parameters, kwargs)
        dict_inputs_axes = map_dictionary(inspect.signature(self.format_axes).parameters, kwargs)
        dict_inputs_lines = map_dictionary(inspect.signature(self.format_lines).parameters, kwargs)
        dict_inputs_legend = map_dictionary(inspect.signature(self.format_legend).parameters, kwargs)

        # set format
        self.format_figure(**dict_inputs_figure)
        self.format_axes(**dict_inputs_axes)
        self.format_lines(**dict_inputs_lines)
        self.format_legend(**dict_inputs_legend)

        self.figure.tight_layout()

    def format_figure(self,
                      figure_size_width=8.,
                      figure_size_height=6.,
                      figure_size_scale=1.,
                      figure_title="",
                      figure_title_font_size=15.):

        self.figure.set_size_inches(w=figure_size_width * figure_size_scale, h=figure_size_height * figure_size_scale)
        self.figure.suptitle(figure_title, fontsize=figure_title_font_size)
        self.figure.set_facecolor((1 / 237., 1 / 237., 1 / 237., 0.0))

    def format_axes(self,
                    axis_label_x="",
                    axis_label_y1="",
                    axis_label_y2="",
                    axis_label_font_size=9.,
                    axis_tick_font_size=8.,
                    axis_lim_x=None,
                    axis_lim_y1=None,
                    axis_lim_y2=None,
                    axis_linewidth=1.,
                    axis_scientific_format_x=False,
                    axis_scientific_format_y1=False,
                    axis_scientific_format_y2=False,
                    axis_tick_width=.5,
                    axis_tick_length=2.5,
                    axis_grid_show=True):
        has_secondary = len(self.axes) > 1

        self.axes[0].set_xlim(axis_lim_x)
        self.axes[0].set_ylim(axis_lim_y1)
        self.axes[1].set_ylim(axis_lim_y2) if has_secondary else None
        self.axes[0].set_xlabel(axis_label_x, fontsize=axis_label_font_size)
        self.axes[0].set_ylabel(axis_label_y1, fontsize=axis_label_font_size)
        self.axes[1].set_ylabel(axis_label_y2, fontsize=axis_label_font_size) if has_secondary else None
        self.axes[0].get_xaxis().get_major_formatter().set_useOffset(axis_scientific_format_x)
        self.axes[0].get_yaxis().get_major_formatter().set_useOffset(axis_scientific_format_y1)
        self.axes[1].get_yaxis().get_major_formatter().set_useOffset(axis_scientific_format_y2) if has_secondary else None

        [i.set_linewidth(axis_linewidth) for i in self.axes[0].spines.values()]
        [i.set_linewidth(axis_linewidth) for i in self.axes[1].spines.values()] if has_secondary else None

        self.axes[0].tick_params(axis='both', which='major', labelsize=axis_tick_font_size, width=axis_tick_width, length=axis_tick_length, direction='in')
        self.axes[0].tick_params(axis='both', which='minor', labelsize=axis_tick_font_size, width=axis_tick_width, length=axis_tick_length, direction='in')
        self.axes[1].tick_params(axis='both', which='major', labelsize=axis_tick_font_size, width=axis_tick_width, length=axis_tick_length, direction='in') if has_secondary else None
        self.axes[1].tick_params(axis='both', which='minor', labelsize=axis_tick_font_size, width=axis_tick_width, length=axis_tick_length, direction='in') if has_secondary else None

        self.axes[0].grid(axis_grid_show, linestyle="--", linewidth=.5, color="black")

        # tick_lines = self.axes[0].get_xticklines() + self.axes[0].get_yticklines()
        # [line.set_linewidth(3) for line in tick_lines]
        #
        # tick_labels = self.axes[0].get_xticklabels() + self.axes[0].get_yticklabels()
        # [label.set_fontsize("medium") for label in tick_labels]

    def format_lines(self,
                     marker_size=3,
                     mark_every=100,
                     marker_fill_style="none",
                     marker_edge_width=.5,
                     line_width=1.,
                     line_style="-"):

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

    def format_legend(self,
                      legend_is_shown=True,
                      legend_loc=0,
                      legend_font_size=8,
                      legend_colour="black",
                      legend_alpha=1.0,
                      legend_is_fancybox=False,
                      legend_line_width=1.):

        line_labels = [l.get_label() for l in self.lines]
        legend = self.axes[len(self.axes) - 1].legend(
            self.lines,
            line_labels,
            loc=legend_loc,
            fancybox=legend_is_fancybox,
            prop={'size': legend_font_size}
            )
        legend.set_visible(legend_is_shown)
        legend.get_frame().set_alpha(legend_alpha)
        legend.get_frame().set_linewidth(legend_line_width)
        legend.get_frame().set_edgecolor(legend_colour)

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
        # todo
        pass

    def remove_line(self, line_name):
        # todo
        pass

    def save_figure(self, name="figure", file_format=".pdf", name_prefix="", name_suffix="", dpi=300):
        time_suffix = False
        str_time = time.strftime("%m%d.%H%M%S")
        if name_suffix == "time":
            name_suffix = str_time
        if name_prefix == "time":
            name_prefix = str_time
        name = "".join([name_prefix, name, name_suffix])
        self.figure.tight_layout()
        name += file_format
        self.figure.savefig(name, bbox_inches='tight', dpi=dpi)

    def show(self):
        self.figure.show(warn=True)


if __name__ == "__main__":
    x = np.arange(0,2*np.pi,0.01)
    y = np.sin(x)
    p = Scatter2D()
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
