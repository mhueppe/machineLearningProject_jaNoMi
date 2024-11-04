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
        Form.resize(483, 564)
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setObjectName(u"gridLayout")
        self.groupBox = QGroupBox(Form)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout_2 = QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.progressBar = QProgressBar(self.groupBox)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setValue(0)

        self.gridLayout_2.addWidget(self.progressBar, 5, 0, 1, 1)

        self.pushButton_enter = QPushButton(self.groupBox)
        self.pushButton_enter.setObjectName(u"pushButton_enter")

        self.gridLayout_2.addWidget(self.pushButton_enter, 5, 2, 1, 1)

        self.textEdit_abstract = QTextEdit(self.groupBox)
        self.textEdit_abstract.setObjectName(u"textEdit_abstract")
        self.textEdit_abstract.setAutoFormatting(QTextEdit.AutoAll)
        self.textEdit_abstract.setLineWrapMode(QTextEdit.WidgetWidth)
        self.textEdit_abstract.setLineWrapColumnOrWidth(22)
        self.textEdit_abstract.setAcceptRichText(True)

        self.gridLayout_2.addWidget(self.textEdit_abstract, 4, 0, 1, 4)

        self.comboBox_modelType = QComboBox(self.groupBox)
        self.comboBox_modelType.setObjectName(u"comboBox_modelType")

        self.gridLayout_2.addWidget(self.comboBox_modelType, 5, 1, 1, 1)


        self.gridLayout.addWidget(self.groupBox, 1, 0, 1, 1)

        self.groupBox_2 = QGroupBox(Form)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.horizontalLayout_2 = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_title = QLabel(self.groupBox_2)
        self.label_title.setObjectName(u"label_title")
        self.label_title.setWordWrap(True)

        self.horizontalLayout_2.addWidget(self.label_title)


        self.gridLayout.addWidget(self.groupBox_2, 0, 0, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.groupBox.setTitle(QCoreApplication.translate("Form", u"Abstract", None))
        self.pushButton_enter.setText(QCoreApplication.translate("Form", u"Generate Title", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Form", u"Title", None))
        self.label_title.setText("")
    # retranslateUi

