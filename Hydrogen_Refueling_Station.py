import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui
import os
from PyQt5.QtCore import QBasicTimer, Qt
import pandas as pd
import pickle
import datetime
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QTabWidget, QSizePolicy)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


buttonData = {
            'TT200':[], 
            'MFM_1':[], 
            'COMP_1':[], 
            'HEX_1':[], 
            'tank':[], 
            'MFM_2':[],
            'CV_1':[], 
            'HEX_2':[],
            'NOZZLE':[], 
            'BUS_1':[]
}


# Hampel initial filter function
def handle_initial_outliers(data, column, n=5, k=3):
    initial_values = data
    median = initial_values.median()
    mad = np.median(np.abs(initial_values - median))
    outlier_threshold = k * mad

    outliers = np.abs(initial_values - median) > outlier_threshold
    if outliers.any():
        non_outlier_median = initial_values[~outliers].median()
        data.loc[data.index[:n], column] = data.loc[data.index[:n], column].mask(outliers, non_outlier_median)

    return data



# Hampel filter function
def hampel_filter(data, window_size=5, k=3):
    # Calculate the rolling median and MAD
    rolling_median = data.rolling(window=window_size, center=True).median()
    
    rolling_mad = data.rolling(window=window_size, center=True).apply(lambda x: np.median(np.abs(x - np.median(x))))
    
    # Identify the outliers
    outliers = np.abs(data - rolling_median) > (k * rolling_mad)
    
    # Replace the outliers using linear interpolation
    data = data.mask(outliers).interpolate()

    return data

# Savgol filter function
def apply_noise_filter(series, window_length=11, polyorder=2):
    sizeP= int(len(series)/4)
    window_length = sizeP if sizeP%2 else sizeP+1
    return savgol_filter(series, window_length, polyorder)

class GraphCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        super(GraphCanvas, self).__init__(fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DataFrame Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()

    def initUI(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.tab_widget = QTabWidget(self)
        layout.addWidget(self.tab_widget)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        self.noise_cancel_button = QPushButton("노이즈 제거", self)
        button_layout.addWidget(self.noise_cancel_button)
        layout.addLayout(button_layout)

        self.noise_cancel_button.clicked.connect(self.apply_noise_cancel)

        self.noise_canceled = False

    def load_dataframe(self, df):
        self.df = df
        self.columns = df.columns
        for col in self.columns:
            graph_canvas = GraphCanvas(self)
            toolbar = NavigationToolbar(graph_canvas, self)
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tab_layout.addWidget(toolbar)
            tab_layout.addWidget(graph_canvas)
            self.tab_widget.addTab(tab, col)
        self.tab_widget.currentChanged.connect(self.update_graph)
        self.update_graph()

    def update_graph(self):
        current_col = self.columns[self.tab_widget.currentIndex()]
        data = self.df[current_col]
        graph_canvas = self.tab_widget.currentWidget().findChild(GraphCanvas)
        graph_canvas.axes.clear()
        graph_canvas.axes.plot(data, 'k', alpha=0.5)

        if self.noise_canceled:
            denoised_data = data.copy()
            denoised_data = handle_initial_outliers(denoised_data, current_col)
            denoised_data = hampel_filter(denoised_data)
            denoised_data = apply_noise_filter(denoised_data)
            graph_canvas.axes.plot(denoised_data, 'b--')
        # y축의 최대값과 최소값을 찾습니다.
        y_min, y_max = graph_canvas.axes.get_ylim()

        # y축의 범위를 5% 위로 확장하고 5% 아래로 축소합니다.
        y_min -= (y_max - y_min) * 0.05
        y_max += (y_max - y_min) * 0.05
        graph_canvas.axes.set_ylim(y_min, y_max)
        graph_canvas.axes.relim()
        graph_canvas.axes.autoscale_view()
        graph_canvas.draw()

    def apply_noise_cancel(self):
        self.noise_canceled = not self.noise_canceled
        self.update_graph()


from_class = uic.loadUiType("Hydrogen Refueling Station.ui")[0]
from_class2 = uic.loadUiType("title.ui")[0]
from functools import partial

input1 = []
data2 = []
step_pick = {}
with open('./configs/step.pkl','rb') as f:
    step_pick = pickle.load(f)

with open("./configs/dataFrame.txt", "r", encoding='UTF8') as f:
    for line in f:
        input1.append(line.replace('\n', ''))

CCC = pd.DataFrame()

#화면을 띄우는데 사용되는 Class 선언
class WindowClass(QMainWindow, from_class):
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.initUI()
        self.initButton()
        self.setButton(['TT200', 'MFM_1', 'COMP_1', 'HEX_1', 'tank', 'MFM_2',
                        'CV_1', 'HEX_2','NOZZLE', 'BUS_1'
                        ])
        # QLabel 및 QPushButton의 위치 및 크기 찾기
        self.save_labels_to_file()
        self.load_labels_from_file()
        self.create_menu()
        self.whoGotSelected()
    
    def extract_labels(self,labelList):
        label_dict = {}
        for child in self.findChildren(QLabel):
            label_dict[child.objectName()] = {'en': child.text()}
        return label_dict
        
    def create_menu(self):
        menubar = self.menuBar()
        help_menu = menubar.addMenu("Help")

        language_action = QAction("Language Selection", self)
        language_action.triggered.connect(self.change_language)
        help_menu.addAction(language_action)

        about_action = QAction("About Software", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def unescape_text(self, text):
        return text.replace("\\n", "\n").replace("\\t", "\t")

    def change_language(self):
        selected_language, ok = QInputDialog.getItem(self, "Language Selection", "Select a language:", ["English", "한국어"], 0, False)

        if ok and selected_language:
            lang_code = "en" if selected_language == "English" else "ko"
            for child in self.findChildren(QLabel):
                if lang_code in self.label_dict[child.objectName()]:
                    child.setText(self.unescape_text(self.label_dict[child.objectName()][lang_code]))
                else:
                    child.setText(self.unescape_text(self.label_dict[child.objectName()]["en"]))



    def save_labels_to_file(self):
        if not os.path.exists("Translation.txt"):
            with open("Translation.txt", "w", encoding="utf-8") as file:
                for child in self.findChildren(QLabel):
                    file.write(f"'{child.objectName()}'\n")
                    file.write("{\n")
                    text2 = child.text().replace("\n", "\\n")
                    file.write(f"\t'en'\t'{text2}'\n")
                    file.write("}\n")

    def load_labels_from_file(self):
        if os.path.exists("Translation.txt"):
            with open("Translation.txt", "r", encoding="utf-8") as file:
                lines = file.readlines()
                # lines = file.strip().split("\n")
                stack = []
                label_dict = {}

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("'") and "{" not in line and stack:
                        parts = line.split('\t')
                        if len(parts) == 2:
                            key, item = parts
                            key = key.strip("'")
                            item = item.strip("'")
                            _, current_dict = stack[-1]
                            current_dict[key] = item
                    elif line.startswith("'"):
                        key = line.strip("'")
                        stack.append((key, {}))
                    elif "}" in line:
                        if stack:
                            key, value = stack.pop()
                            if stack:
                                _, parent = stack[-1]
                                parent[key] = value
                            else:
                                label_dict[key] = value

                if stack:
                    key, value = stack.pop()
                    label_dict[key] = value

                self.label_dict = label_dict

    def show_about(self):
        AboutWindow = from_class2
        about_dialog = QDialog(self)
        about_ui = AboutWindow()
        about_ui.setupUi(about_dialog)
        about_dialog.exec_()

    def setButton(self, labelList):
        for label_name in labelList:
            label = self.findChild(QLabel, label_name)

            if label:
                # Create a transparent button with no shadow effect
                button = QPushButton(self)
                button.setObjectName(f"{label_name}Button")
                button.setGeometry(label.geometry())
                button.setStyleSheet("QPushButton { background-color: transparent; border: none; }")
                button.clicked.connect(self.onButtonClicked)
                button.raise_()
                button.show()
            else:
                print(f"QLabel '{label_name}' not found")

    def onButtonClicked(self):
        # Get the sender (clicked button) and its object name
        sender = self.sender()
        button_name = sender.objectName()

        print(f"Clicked button: {button_name}")

        self.main_app = App()

        # Create example DataFrame
        example_data = pd.DataFrame(np.random.rand(100, 3), columns=["A", "B", "C"])

        self.main_app.load_dataframe(example_data)
        self.main_app.show()
        


    #타이머 이벤트 함수
    def timerEvent(self, e):
        updateGUI(self)
        self.fstep += self.SPEED / 10
        self.step = int(self.fstep)

    def initUI(self):
        #타이머 생성
        self.timer = QBasicTimer()
        self.step = 0
        self.SPEED = 1
        self.fstep = 0

        chillerBody = [self.CH_1, self.CH_2, self.CH_3, self.CH_4]
        changeColor(chillerBody, 3)
        changeColor([self.VENT], 1)

        # Adding tool bar to the window
        toolBar = QToolBar("Main toolbar")
        self.addToolBar(toolBar)

        # Creating Items for the tool-bar
        folderOpen = QAction(QtGui.QIcon("./image/folder.png"), "Open Folder", toolBar)
        folderOpen.triggered.connect(self.whoGotSelected)
        folderOpen.setStatusTip("Open Folder")
        toolBar.addAction(folderOpen)
        
        toolBar.addSeparator()
        # Adding those item into the tool-bar
        screenShot = QAction(QtGui.QIcon("./image/screen.png"), "Screen shot", toolBar) # screenshot
        screenShot.triggered.connect(self.whoGotSelected2)
        screenShot.setStatusTip("Screen shot")
        toolBar.addAction(screenShot)

        self.show()
    
    #버튼에 기능을 연결
    def initButton(self):        
        self.T.clicked.connect(self.CalculatFunction)
        self.RESET.clicked.connect(self.CalculatFunction2)

    def whoGotSelected2(self):
        directory ='./screenshot/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        date = datetime.datetime.now()
        filename = date.strftime('./screenshot/%Y-%m-%d(%H-%M-%S).png') # 파일이름 만들기용도
        p = QtGui.QScreen.grabWindow(app.primaryScreen(),self.screen_1.winId())#(메인화면, 현재위젯)
        p.save(filename, 'png')
        # QScreen은 PYQT5.QtGui에 포함되어 있는 항목으로 grabwindow로 캡쳐가 가능합니다.

    def whoGotSelected(self):
        fname=QFileDialog.getOpenFileName(self,'Open file','./data/')
        if fname[0].endswith('.xlsx'):
            self.menuBar().clear()
            data = pd.read_excel(fname[0],sheet_name=None)
            sheets = data.keys()
            resetTime(self)
            for i in sheets:
                data3 = pd.read_excel(fname[0],sheet_name=i)
                i = int(i)
                data2.append(data3)
                loadfile = QAction('%d'%(i), self)
                loadfile.triggered.connect(partial(self.add_open, i))
                self.menuBar().addAction(loadfile)
        
        elif fname[0].endswith('.csv'):
            self.menuBar().clear()
            global CCC
            CCC = pd.read_csv(fname[0])
            resetTime(self)
            c = step_pick[fname[0].split('/')[-1][0:13]]
            for k in c:
                d = CCC.index[CCC['시각'] == c[k][0]].to_list()[0]
                loadfile = QAction(k, self)
                loadfile.triggered.connect(partial(self.add_open2, d))
                self.menuBar().addAction(loadfile)
        self.create_menu()

    def add_open(self, num):
        global CCC
        CCC = data2[num-1]

        resetTime(self)

    def add_open2(self, num):
        self.step = num
        self.fstep = num
        updateGUI(self)

    #버튼 함수 정리
    def CalculatFunction(self):
        if self.step > len(CCC):
            resetTime(self)
        if self.timer.isActive():
            self.timer.stop()
            self.T.setText('Start')
        else:
            self.timer.start(100, self)
            self.T.setText('Stop')
    
    def CalculatFunction2(self):
        self.SPEED = 1
        resetTime(self)
        
    
    #키가 눌러졌을 때 실행됨
    def keyPressEvent(self, e): 
        if e.key() == Qt.Key_R:
            resetTime(self)
        elif e.key() == Qt.Key_A:
            controlSkip(self, -1)
        elif e.key() == Qt.Key_D:
            controlSkip(self, 1)
        elif e.key() == Qt.Key_W:
            controlSpeed(self, 1)
        elif e.key() == Qt.Key_S:
            controlSpeed(self, -1)

def resetTime(self):
    self.step = 0
    self.fstep = 0
    updateGUI(self)

#control skip
def controlSkip(self, var):
    time1 = self.step
    if(time1>=len(CCC)):
        self.step = len(CCC) - 1

    self.step = self.step + var * int(self.SPEED)
    self.fstep = self.step
    updateGUI(self)

#색상변경
def RainbowColor(pressure):
    R = 0
    G = 0
    B = 0
    p = (pressure)/ (800)
    
    if(p < 0):
        B = 255
    elif(p >= 1):
        R = 255
    elif(p>=0 and p < 0.125):
        G = int(255 * (8*(p - 0)))
        B = 255
    elif(p>=0.125 and p < 0.25):
        G = 255
        B = 255 - int(255 * (8*(p - 0.125)))
    elif(p>=0.25 and p < 0.5):
        R = int(255 * (4*(p - 0.25)))
        G = 255
    elif(p>=0.5 and p < 1):
        R = 255
        G = 255 - int(255 * (2*(p -0.5)))

    return (R,G,B)

#재생속도 조절
def controlSpeed(self, var):
    SPEED = self.SPEED
    if SPEED <= 1:
        if var > 0:
            self.SPEED += var
    else: self.SPEED += var
    self.SPEED_T.setText("%d"%(self.SPEED))
    
#색상 조절
def changeColor(bl, pp):
    R, G, B = RainbowColor(pp)
    for bi in bl:
        bi.setStyleSheet("background-color: rgb(%d, %d, %d);"%(R, G, B))

#Bank PrograssBar Change
def changeProgassBar(b1, b2, b3, pp, size,text):
    ew = int(pp/size*100)
    if ew >100:
        ew = 100
    b1.setText("%s Bank \n(%d %%) %0.1f bar"%(text, ew, pp))
    b2.setText("%0.1f"%pp)
    b3.setValue(ew)

#updateGui
def updateGUI(self):
    time1 = self.step
    if(time1>=len(CCC)):
        self.step = len(CCC) - 1
        time1 = len(CCC) - 1
    elif time1 < 0:
        self.step = 0
        time1 = 0
    #variable
    trailerBody = [self.TRAILER_1,self.TRAILER_2,self.TRAILER_3,self.TRAILER_4,]
    compBody = [self.PI_1, self.PI_2, self.PI_3]
    hexBody = [self.PO_1, self.PO_2, self.PO_3, self.PO_4, self.PO_5, self.PO_6]
    highBody = [self.HIGH_1, self.HIGH_2, self.HIGH_3]
    midBody = [self.MID_1, self.MID_2, self.MID_3]
    lowhBody = [self.LOW_1, self.LOW_2, self.LOW_3]
    mfm2Body = [self.C_1, self.C_2, self.C_3, self.C_4, self.C_5, self.C_6]
    nozzleBody = [self.NOZZLE_1, self.NOZZLE_2, self.NOZZLE_3, self.NOZZLE_4]

    #TT200
    changeColor(trailerBody, CCC[input1[0]][time1])
    self.TT200_TEXT.setText("%0.1f"%(CCC[input1[0]][time1]))
    
    #MFM1
    self.MFM1_M_1.setText("%0.1f"%(float(CCC[input1[1]][time1])* 1000 / 60))
    self.MFM1_M_2.setText("%d"%(float(CCC[input1[25]][time1])))

    #COMIN
    changeColor(compBody, CCC[input1[2]][time1])
    self.COMIN_P.setText("%0.1f"%(CCC[input1[2]][time1]))
    self.COMIN_T.setText("%0.1f"%(CCC[input1[3]][time1]))
    
    #COMOUT
    self.COMOUT_T.setText("%0.1f"%(CCC[input1[4]][time1]))
    
    #HEXOUT
    changeColor(hexBody, CCC[input1[5]][time1])
    self.HEXOUT_P.setText("%0.1f"%(CCC[input1[5]][time1]))
    self.HEXOUT_T.setText("%0.1f"%(CCC[input1[6]][time1]))
    
    #HIGH
    changeColor(highBody, CCC[input1[7]][time1])
    changeProgassBar(self.HIGHBANK_TEXT, self.HIGH_P , self.HIGHBANK_PRO,CCC[input1[7]][time1], 875, 'HIGH')
    
    #MID
    changeColor(midBody, CCC[input1[8]][time1])
    changeProgassBar(self.MIDBANK_TEXT, self.MID_P , self.MIDBANK_PRO,CCC[input1[8]][time1], 450, 'MID')
    
    #LOW
    changeColor(lowhBody, CCC[input1[9]][time1])
    changeProgassBar(self.LOWBANK_TEXT, self.LOW_P , self.LOWBANK_PRO,CCC[input1[9]][time1], 450, 'LOW')

    #MFM2
    changeColor(mfm2Body, CCC[input1[10]][time1])
    self.MFM2_P.setText("%0.1f"%(CCC[input1[10]][time1]))
    self.MFM2_M.setText("%0.2f"%(float(CCC[input1[11]][time1]) * 1000 / 60))

    #CV
    self.CV_C.setText("%d"%(CCC[input1[12]][time1]))
    self.CV_PRO.setValue(CCC[input1[12]][time1])

    #PRE-COOL
    self.HEX2_T.setText("%0.1f"%(CCC[input1[13]][time1]))

    #NOZZLE
    changeColor(nozzleBody, CCC[input1[14]][time1])
    self.NOZZLE_P.setText("%0.1f"%(CCC[input1[14]][time1]))
    self.NOZZLE_T.setText("%0.1f"%(CCC[input1[15]][time1]))

    #차량
    changeColor([self.BUSTANK_1], CCC[input1[16]][time1])
    self.V_P.setText("%d"%(CCC[input1[16]][time1]))
    self.V_T.setText("%0.1f"%(CCC[input1[17]][time1]))
    self.V_SOC.setText("%d"%(CCC[input1[18]][time1]))
    self.V_M.setText("%0.2f"%(CCC[input1[18]][time1]/100*40.2*880/1000))

    #주변온도
    self.A_T.setText("%0.1f"%(CCC[input1[19]][time1]))

    #패널
    self.B_1.setStyleSheet("background-color: rgb(%d, %d, %d);"%((0, 255, 0) if CCC[input1[20]][time1] else (255, 0, 0) ))
    self.B_2.setStyleSheet("background-color: rgb(%d, %d, %d);"%((0, 255, 0) if CCC[input1[21]][time1] else (255, 0, 0) ))
    self.B_3.setStyleSheet("background-color: rgb(%d, %d, %d);"%((0, 255, 0) if CCC[input1[22]][time1] else (255, 0, 0) ))
    self.B_4.setStyleSheet("background-color: rgb(%d, %d, %d);"%((0, 255, 0) if CCC[input1[23]][time1] else (255, 0, 0) ))
    
    #시간
    Y, M, D, T = CCC[input1[24]][time1].split()
    self.TIME_1.setText("%s-%s-%s\n%s"%(Y,M,D,T))
    self.TIME_T.setText("%d / %d"%(int(time1)+1,len(CCC)))

    #디스펜서
    self.P_i.setText("%d"%(CCC[input1[26]][time1]*10))
    self.P_t.setText("%d"%(CCC[input1[27]][time1]*10))
    
    #속도
    self.SPEED_T.setText("%d"%(self.SPEED))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()

    app.exec_()