# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'interface.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(400, 300)
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setObjectName(u"gridLayout")
        self.groupBox = QGroupBox(Form)
        self.groupBox.setObjectName(u"groupBox")
        self.horizontalLayout = QHBoxLayout(self.groupBox)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.lineEdit_input = QLineEdit(self.groupBox)
        self.lineEdit_input.setObjectName(u"lineEdit_input")

        self.horizontalLayout.addWidget(self.lineEdit_input)

        self.pushButton_enter = QPushButton(self.groupBox)
        self.pushButton_enter.setObjectName(u"pushButton_enter")
        sizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_enter.sizePolicy().hasHeightForWidth())
        self.pushButton_enter.setSizePolicy(sizePolicy)

        self.horizontalLayout.addWidget(self.pushButton_enter)


        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)

        self.groupBox_2 = QGroupBox(Form)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.horizontalLayout_2 = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.textBrowser_output = QTextBrowser(self.groupBox_2)
        self.textBrowser_output.setObjectName(u"textBrowser_output")

        self.horizontalLayout_2.addWidget(self.textBrowser_output)


        self.gridLayout.addWidget(self.groupBox_2, 1, 0, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.groupBox.setTitle(QCoreApplication.translate("Form", u"Input", None))
        self.pushButton_enter.setText(QCoreApplication.translate("Form", u"Enter", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Form", u"Output", None))
    # retranslateUi

