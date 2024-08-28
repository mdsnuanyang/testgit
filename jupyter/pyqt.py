import sys

from PyQt5.QtWidgets import QApplication , QWidget,QPushButton,QLabel,QLineEdit
if __name__ == '__main__':
    app = QApplication(sys.argv)#创建·对象
    w = QWidget()
    w.setWindowTitle("第一个pyqt")

    btn = QPushButton("按钮")
    btn.setParent(w)
    label = QLabel("账号",w)
    label.setGeometry(20,20,30,30)
    edit = QLineEdit(w)
    edit.setPlaceholderText("请输入账号")
    edit.setGeometry(55,20,200,20)
    w.resize(300,300)
    w.show()
    app.exec_()  