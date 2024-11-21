from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLabel
from PyQt5.QtGui import QPixmap
from importlib import import_module

class Ui_Dialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.classifier = import_module('VGG_case1')  # VGG 분류기 모듈 로드

    def setupUi(self):
        self.setObjectName("Dialog")
        self.resize(766, 670)

        self.graphicsView = QtWidgets.QGraphicsView(self)
        self.graphicsView.setGeometry(QtCore.QRect(25, 20, 721, 491))
        self.graphicsView.setSizeIncrement(QtCore.QSize(0, 0))
        self.graphicsView.setFrameShadow(QtWidgets.QFrame.Raised)
        self.graphicsView.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.graphicsView.setAlignment(QtCore.Qt.AlignJustify | QtCore.Qt.AlignVCenter)
        self.graphicsView.setObjectName("graphicsView")

        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setGeometry(QtCore.QRect(670, 520, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Load image")  # 버튼 텍스트 변경
        self.pushButton.clicked.connect(self.open_image)  # 버튼 클릭 시 open_image 호출

        self.resultLabel = QLabel(self)
        self.resultLabel.setGeometry(QtCore.QRect(25, 530, 721, 100))
        self.resultLabel.setWordWrap(True)
        self.resultLabel.setObjectName("resultLabel")

        self.setWindowTitle("AI Image Classifier")

    def open_image(self):
        # 파일 다이얼로그를 열어 이미지 파일 선택
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        
        if file_path:
            self.display_image(file_path)  # 이미지를 표시하는 함수 호출
            self.classify_image(file_path)  # 이미지 분류 함수 호출

    def display_image(self, file_path):
        # 이미지를 QGraphicsView에 표시
        scene = QtWidgets.QGraphicsScene()
        pixmap = QPixmap(file_path)
        
        if pixmap.isNull():
            QMessageBox.warning(self, "Error", "Could not load image.")
            return
        
        # 원하는 높이 설정
        desired_height = 491
        # 비율에 맞춰 크기 조정
        scaled_pixmap = pixmap.scaledToHeight(desired_height, QtCore.Qt.SmoothTransformation)

        scene.addPixmap(scaled_pixmap)
        self.graphicsView.setScene(scene)

    def classify_image(self, file_path):
        # VGG.py의 classify_image 함수 호출
        try:
            results = self.classifier.classify_image(file_path)  # 분류 결과 가져오기
            result_text = "\n".join(results)  # 결과 텍스트 형식
            self.resultLabel.setText(result_text)  # 결과 레이블에 출력
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not classify image: {str(e)}")

# 메인 함수
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_Dialog()
    ui.show()
    sys.exit(app.exec_())
