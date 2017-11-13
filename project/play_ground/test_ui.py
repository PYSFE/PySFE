# -*- coding:utf-8 -*-
import time
from FireDynamics.FurnaceModel import main_function
import sys
import os
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMessageBox
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)


path = os.path.dirname(os.path.abspath(__file__))
ui_file = "UI/gui_main.ui"
Ui_MainWindow, Ui_QtBaseClass = uic.loadUiType(os.path.join(path, ui_file))


class Application(Ui_QtBaseClass, Ui_MainWindow):
    def __init__(self, parent=None):
        Ui_QtBaseClass.__init__(self, parent)
        self._figure = None

        self.setupUi(self)
        self.update_parameters(self.static_default_parameters())
        self.show()

        self.on_click_submit()  # load graph based on default values

        # (figure_width, figure_height) = tuple(self._figure.figure.get_size_inches)
        # self.update_parameters(figure_size_width = figure_width, figure_size_height = figure_height, figure_size_scale = 1.0)

        # connect button to function on_click
        self.pushButton_submit.clicked.connect(self.on_click_submit)
        self.pushButton_quit.clicked.connect(self.on_click_quit)
        self.pushButton_update.clicked.connect(self.on_click_update)
        self.pushButton_save.clicked.connect(self.on_click_save)

    def on_click_submit(self):
        p = self.get_parameters()
        print("PARSED PARAMETERS ARE:")
        for i in p:
            print(i + ': ' + str(p[i]))
        time.sleep(0.1)

        self._figure = main_function(
            p['time_step'], p['time_cap'], p['gas_emissivity'], p['vent_mass_flow_rate'],
            p['specimen_exposed_area'], p['specimen_volume'], p['specimen_heat_of_combustion'], p['specimen_burning_rate'], p['specimen_density'],
            p['lining_surface_emissivity'], p['lining_surface_area'], p['lining_thickness'], p['lining_thermal_conductivity'], p['lining_density'], p['lining_specific_heat']
        )
        self.add_mpl(self._figure.figure)
        self.on_click_save()
        return self.get_parameters()

    def on_click_update(self):
        self.remove_mpl()
        self.on_click_submit()
        self.on_click_save()

    def on_click_quit(self):
        self.close()

    def on_click_save(self):
        old_size = self._figure.figure.get_size_inches().tolist()

        p = self.get_parameters()
        new_size = [p['figure_size_width']*p['figure_size_scale'], p['figure_size_height']*p['figure_size_scale']]

        self._figure.figure.set_size_inches(new_size)
        self._figure.save_figure("output")
        print("Figure saved at: " + os.getcwd())

        self._figure.figure.set_size_inches(old_size)
        self.canvas.draw()

    def add_mpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas,
                                         self.mplwindow, coordinates=True)
        self.mplvl.addWidget(self.toolbar)

    def remove_mpl(self):
        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()

    def update_parameters(self, parameters):
        for key in parameters:
            parameters[key] = str(parameters[key])

        self.lineEdit_timeStep.setText(parameters['time_step'])
        self.lineEdit_timeCap.setText(parameters['time_cap'])
        self.lineEdit_gasEmissivity.setText(parameters['gas_emissivity'])
        self.lineEdit_ventMassFlowRate.setText(parameters['vent_mass_flow_rate'])

        self.lineEdit_specimenExposedArea.setText(parameters['specimen_exposed_area'])
        self.lineEdit_specimenVolume.setText(parameters['specimen_volume'])
        self.lineEdit_specimenHeatOfCombustion.setText(parameters['specimen_heat_of_combustion'])
        self.lineEdit_specimenBurningRate.setText(parameters['specimen_burning_rate'])
        self.lineEdit_specimenDensity.setText(parameters['specimen_density'])

        self.lineEdit_liningSurfaceEmissivity.setText(parameters['lining_surface_emissivity'])
        self.lineEdit_liningSurfaceArea.setText(parameters['lining_surface_area'])
        self.lineEdit_liningThickness.setText(parameters['lining_thickness'])
        self.lineEdit_liningThermalConductivity.setText(parameters['lining_thermal_conductivity'])
        self.lineEdit_liningDensity.setText(parameters['lining_density'])
        self.lineEdit_liningSpecificHeat.setText(parameters['lining_specific_heat'])

        self.lineEdit_figure_size_width.setText(parameters['figure_size_width'])
        self.lineEdit_figure_size_height.setText(parameters['figure_size_height'])
        self.lineEdit_figure_size_scale.setText(parameters['figure_size_scale'])
        pass

    def get_parameters(self):
        # self.lineEdit_timeStep.text()
        # self.lineEdit_timeCap.text()
        # self.lineEdit_gasEmissivity.text()
        # self.lineEdit_ventMassFlowRate.text()
        #
        # self.lineEdit_specimenExposedArea.text()
        # self.lineEdit_specimenVolume.text()
        # self.lineEdit_specimenHeatOfCombustion.text()
        # self.lineEdit_specimenBurningRate.text()
        # self.lineEdit_specimenDensity.text()
        #
        # self.lineEdit_liningSurfaceEmissivity.text()
        # self.lineEdit_liningSurfaceArea.text()
        # self.lineEdit_liningThickness.text()
        # self.lineEdit_liningThermalConductivity.text()
        # self.lineEdit_liningDensity.text()
        # self.lineEdit_liningSpecificHeat.text()

        parameters={
            "time_step": self.lineEdit_timeStep.text(),
            "time_cap": self.lineEdit_timeCap.text(),
            'gas_emissivity': self.lineEdit_gasEmissivity.text(),
            'vent_mass_flow_rate': self.lineEdit_ventMassFlowRate.text(),

            'specimen_exposed_area': self.lineEdit_specimenExposedArea.text(),
            'specimen_volume': self.lineEdit_specimenVolume.text(),
            'specimen_heat_of_combustion': self.lineEdit_specimenHeatOfCombustion.text(),
            'specimen_burning_rate': self.lineEdit_specimenBurningRate.text(),
            'specimen_density': self.lineEdit_specimenDensity.text(),

            'lining_surface_emissivity': self.lineEdit_liningSurfaceEmissivity.text(),
            'lining_surface_area': self.lineEdit_liningSurfaceArea.text(),
            'lining_thickness': self.lineEdit_liningThickness.text(),
            'lining_thermal_conductivity': self.lineEdit_liningThermalConductivity.text(),
            'lining_density': self.lineEdit_liningDensity.text(),
            'lining_specific_heat': self.lineEdit_liningSpecificHeat.text(),

            'figure_size_width': self.lineEdit_figure_size_width.text(),
            'figure_size_height': self.lineEdit_figure_size_height.text(),
            'figure_size_scale': self.lineEdit_figure_size_scale.text(),
        }

        for item in parameters:
            parameters[item] = float(eval(parameters[item]))

        return parameters

    def run_simulation(self):
        d=1

    @staticmethod
    def static_default_parameters():
        parameters = {
            'time_step': 0.5,
            'time_cap': '60*60*2',
            'gas_emissivity': 0.7,
            'vent_mass_flow_rate': 0.0735,

            'specimen_exposed_area': 1,
            'specimen_volume': 0.4,
            'specimen_heat_of_combustion': '19e6',
            'specimen_burning_rate': '0.7/1000./60.',
            'specimen_density': 640,

            'lining_surface_emissivity': 0.7,
            'lining_surface_area': '(4*3+3*1.5+1.5*4)*2',
            'lining_thickness': .05,
            'lining_thermal_conductivity': 1.08997,
            'lining_density': 1700,
            'lining_specific_heat': 820,

            'figure_size_width': 10,
            'figure_size_height': 7.5,
            'figure_size_scale': 0.6,
        }
        return parameters


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Application()
    sys.exit(app.exec_())  # exit python when app is finished
