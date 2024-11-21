from image_viewer import *
from PyQt5.QtWidgets import QApplication, QDialog
import sys


class My_Application(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.pushButton.clicked.connect(self.open_image)  # 버튼 클릭 시 open_image 호출

    def open_image(self):
        file_path = self.ui.open_image()  # 선택한 파일 경로 가져오기
        if file_path:
            self.checkPath(file_path)  # 경로를 checkPath 함수에 전달

    def checkPath(self, image_path):
        if image_path:
            # 이미지를 QGraphicsView에 표시하는 로직은 open_image에서 처리됨
            print(f"Selected image path: {image_path}")  # 선택한 파일 경로 출력

if __name__ == '__main__':
    app = QApplication(sys.argv)
    class_instance = My_Application()
    class_instance.show()
    sys.exit(app.exec_())
