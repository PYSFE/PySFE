import sys
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QProgressBar
import time
from PyQt5.QtCore import (QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal)


class ProgressBar(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "title"
        self.window_width = 500
        self.window_height = 200
        self.pos_width = 50
        self.pos_height = 50
        self.text_status = "status"
        self.progress_bar_obj = QProgressBar(self)

        self.initiate_ui()

    def initiate_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.pos_width, self.pos_height, self.window_width, self.window_height)
        self.statusBar().showMessage(self.text_status)
        self.progress_bar_obj.setGeometry(50, 0, 400, 100)
        self.progress_bar_obj.setValue(50)
        self.show()

    def update_progress(self, v=0):
        self.progress_bar_obj.setValue(v)

    def start_thread(self):
        # Create the new thread. The target function is 'myThread'. The
        # function we created in the beginning.
        pass


# Subclassing QThread
# http://qt-project.org/doc/latest/qthread.html
class AThread(QThread):

    def run(self):
        count = 0
        while count < 5:
            time.sleep(1)
            print("A Increasing")
            count += 1



# Subclassing QObject and using moveToThread
# http://blog.qt.digia.com/blog/2007/07/05/qthreads-no-longer-abstract
class SomeObject(QObject):

    finished = pyqtSignal()

    def long_running(self):
        count = 0
        while count < 5:
            time.sleep(1)
            print("B Increasing")
            count += 1
        self.finished.emit()


# Using a QRunnable
# http://qt-project.org/doc/latest/qthreadpool.html
# Note that a QRunnable isn'c a subclass of QObject and therefore does
# not provide signals and slots.
class Runnable(QRunnable):

    def run(self):
        count = 0
        app = QCoreApplication.instance()
        while count < 5:
            print("C Increasing")
            time.sleep(1)
            count += 1
        app.quit()


def using_q_thread():
    app = QCoreApplication([])
    thread = AThread()
    thread.finished.connect(app.exit)
    thread.start()
    sys.exit(app.exec_())


def using_move_to_thread():
    app = QCoreApplication([])
    objThread = QThread()
    obj = SomeObject()
    obj.moveToThread(objThread)
    obj.finished.connect(objThread.quit)
    objThread.started.connect(obj.long_running)
    objThread.finished.connect(app.exit)
    objThread.start()
    sys.exit(app.exec_())


def using_q_runnable():
    app = QCoreApplication([])
    runnable = Runnable()
    QThreadPool.globalInstance().start(runnable)
    sys.exit(app.exec_())


if __name__ == "__main__":
    using_q_thread()
    # using_move_to_thread()
    # using_q_runnable()

    App = QApplication(sys.argv)
    ex = ProgressBar()
    sys.exit(App.exec_())
