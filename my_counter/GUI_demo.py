import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel,QFileDialog,QWidget
from PyQt5 import QtGui
from my_counter.Ui_myUI import Ui_MainWindow
from my_counter.counter_new import detect
# from my_counter.counter2 import detect
# from my_counter.counter10b import detect
from PIL import Image
import cv2

class MainWin(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setConnect()
        self.imgNamepath=""
        self.img=None
        self.thr=None
        self.cutPosition=[[0,0],[720,1280]]
        self.filewidget=None
        self.results=None
        self.detect_img=None

    def setConnect(self):
        '''绑定槽函数'''
        self.Button_imginput.clicked.connect(self.imgInput)
        self.Button_detect.clicked.connect(self.detect)

    
    def imgInput(self):
        '''选择读取图片,在窗口展示720*1280'''
        self.imgNamepath, imgType = QFileDialog.getOpenFileName(self.centralwidget, "选择图片", "E:\\","*.jpg;;*.png")
        self.img = QtGui.QPixmap(self.imgNamepath)#.scaled(self.label_show.width(), self.label_show.height())
        self.label_show.setPixmap(self.img)
        self.scrollArea.setWidget(self.label_show)
        # self.listWidget.setPixmap(self.img)
        # self.filewidget = QWidget()
        # self.filewidget.setMinimumSize(720, 1280)
        # lab = QLabel(self.filewidget)
        # lab.setFixedSize(720,1280)
        # lab.setPixmap(self.img)
        # self.filewidget = QWidget(lab)
        # self.scrollArea.setWidget(self.filewidget)


    def imgCut(self):
        '''选择感兴趣区域'''
        pass

    def detect(self):
        '''检测感兴趣区域计数并画图,处理时的图片尺寸是1440*2560,并把检测结果保存'''
        self.results,self.detect_img=detect(self.imgNamepath)
        # self.detect_img= Image.fromarray(self.detect_img).toqpixmap()
        RGBImg=cv2.cvtColor(self.detect_img,cv2.COLOR_BGR2RGB)
        # self.detect_img=QtGui.QImage(RGBImg,RGBImg.shape[1],RGBImg.shape[0],QtGui.QImage.Format.Format_RGB888)        
        self.detect_img =QtGui.QPixmap(QtGui.QImage(RGBImg,RGBImg.shape[1],RGBImg.shape[0],RGBImg.shape[2]*RGBImg.shape[1],QtGui.QImage.Format.Format_RGB888))
        self.label_show.setPixmap(self.detect_img)
        self.scrollArea.setWidget(self.label_show)
        count=0
        for result in self.results:
            count+=len(result)
        self.lcdNumber_result.display(count-1)
        # self.label_show.setPixmap(self.detect_img.resize(self.label_show.width(), self.label_show.height()))

    def setArg(self):
        '''自定义检测阈值'''

    def save(self):
        '''保存检测图片'''    
        img = self.label_show.pixmap().toImage()
        fpath, ftype = QFileDialog.getSaveFileName(self.centralwidget, "保存图片", "d:\\", "*.jpg;;*.png")
        img.save(fpath)

    def exit():
        '''退出程序'''
        # app.quit()
        pass


if __name__ == "__main__": 
    # 创建一个app主体
    app = QApplication(sys.argv)
    # 创建一个主窗口 
    win = MainWin()
    # 显示程序窗口
    win.show()
    # 启动主循环，开始程序的运行
    sys.exit(app.exec_())

