# FileName : PyQtDemo.py
# Author  : Adil
# DateTime : 2018/2/1 11:07
# SoftWare : PyCharm


'''
from PyQt5 import QtWidgets, QtGui
import sys

app = QtWidgets.QApplication(sys.argv)
window = QtWidgets.QWidget();
window.show()
sys.exit(app.exec_())

'''
import sys
import UI.openpose
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
if __name__ == '__main__':
    myapp = QtWidgets.QApplication(sys.argv)
    mymian = QtWidgets.QMainWindow()
    myUI = UI.openpose.Ui_MainWindow()
    myUI.setupUi(mymian)
    mymian.show()
    sys.exit(myapp.exec_())


'''
import sys
import UI.untitled # 导入转换完成的文件
from PyQt5.QtWidgets import QApplication, QDialog
if __name__ == '__main__':
  myapp = QApplication(sys.argv)
  myDlg = QDialog()
  myUI = UI.untitled.Ui_Dialog()
  myUI.setupUi(myDlg)
  myDlg.show()
  sys.exit(myapp.exec_())
'''
'''
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()

        self.resize(600, 400)
        self.setWindowTitle("label显示图片")

        self.label = QLabel(self)
        self.label.setText("   显示图片")
        self.label.setFixedSize(300, 200)
        self.label.move(160, 160)

        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )

        btn = QPushButton(self)
        btn.setText("打开图片")
        btn.move(10, 30)
        btn.clicked.connect(self.openimage)

        
    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    sys.exit(app.exec_())
'''

