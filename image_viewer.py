from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(766, 569)

        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(25, 20, 721, 491))
        self.graphicsView.setSizeIncrement(QtCore.QSize(0, 0))
        self.graphicsView.setFrameShadow(QtWidgets.QFrame.Raised)
        self.graphicsView.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.graphicsView.setAlignment(QtCore.Qt.AlignJustify | QtCore.Qt.AlignVCenter)
        self.graphicsView.setObjectName("graphicsView")

        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(670, 520, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Load image")  # 버튼 텍스트 변경
        self.pushButton.clicked.connect(self.open_image)  # 버튼 클릭 시 open_image 호출

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Image Viewer"))

    def open_image(self):
        # 파일 다이얼로그를 열어 이미지 파일 선택
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        
        if file_path:
            self.display_image(file_path)  # 이미지를 표시하는 함수 호출
            return file_path  # 선택한 파일 경로 반환

    def display_image(self, file_path):
        # 이미지를 QGraphicsView에 표시
        scene = QtWidgets.QGraphicsScene()
        pixmap = QPixmap(file_path)
        
        if pixmap.isNull():
            QMessageBox.warning(None, "Error", "Could not load image.")
            return
        
        scene.addPixmap(pixmap)
        self.graphicsView.setScene(scene)

# 메인 함수
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
