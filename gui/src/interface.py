# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'interface.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(602, 750)
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setObjectName(u"gridLayout")
        self.groupBox_2 = QGroupBox(Form)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.horizontalLayout_2 = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.listWidget_titles = QListWidget(self.groupBox_2)
        self.listWidget_titles.setObjectName(u"listWidget_titles")

        self.horizontalLayout_2.addWidget(self.listWidget_titles)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.pushButton_stop = QPushButton(self.groupBox_2)
        self.pushButton_stop.setObjectName(u"pushButton_stop")

        self.verticalLayout.addWidget(self.pushButton_stop)


        self.horizontalLayout_2.addLayout(self.verticalLayout)

        self.label_title = QLabel(self.groupBox_2)
        self.label_title.setObjectName(u"label_title")
        self.label_title.setWordWrap(True)

        self.horizontalLayout_2.addWidget(self.label_title)


        self.gridLayout.addWidget(self.groupBox_2, 0, 0, 1, 1)

        self.groupBox = QGroupBox(Form)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout_2 = QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.horizontalSlider_temperature = QSlider(self.groupBox)
        self.horizontalSlider_temperature.setObjectName(u"horizontalSlider_temperature")
        self.horizontalSlider_temperature.setMinimum(1)
        self.horizontalSlider_temperature.setMaximum(200)
        self.horizontalSlider_temperature.setOrientation(Qt.Horizontal)

        self.gridLayout_2.addWidget(self.horizontalSlider_temperature, 6, 0, 1, 2)

        self.spinBox_number_titles = QSpinBox(self.groupBox)
        self.spinBox_number_titles.setObjectName(u"spinBox_number_titles")
        self.spinBox_number_titles.setMinimum(1)
        self.spinBox_number_titles.setMaximum(16)

        self.gridLayout_2.addWidget(self.spinBox_number_titles, 6, 3, 1, 1)

        self.label_temperature = QLabel(self.groupBox)
        self.label_temperature.setObjectName(u"label_temperature")

        self.gridLayout_2.addWidget(self.label_temperature, 5, 0, 1, 1)

        self.progressBar = QProgressBar(self.groupBox)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setValue(0)

        self.gridLayout_2.addWidget(self.progressBar, 7, 0, 1, 1)

        self.label_number_titles = QLabel(self.groupBox)
        self.label_number_titles.setObjectName(u"label_number_titles")

        self.gridLayout_2.addWidget(self.label_number_titles, 5, 3, 1, 1)

        self.spinBox_temperature = QSpinBox(self.groupBox)
        self.spinBox_temperature.setObjectName(u"spinBox_temperature")
        self.spinBox_temperature.setMinimum(1)
        self.spinBox_temperature.setMaximum(200)

        self.gridLayout_2.addWidget(self.spinBox_temperature, 6, 2, 1, 1)

        self.comboBox_modelType = QComboBox(self.groupBox)
        self.comboBox_modelType.setObjectName(u"comboBox_modelType")

        self.gridLayout_2.addWidget(self.comboBox_modelType, 7, 2, 1, 1)

        self.pushButton_enter = QPushButton(self.groupBox)
        self.pushButton_enter.setObjectName(u"pushButton_enter")

        self.gridLayout_2.addWidget(self.pushButton_enter, 7, 3, 1, 1)

        self.textBrowser_abstract = QTextBrowser(self.groupBox)
        self.textBrowser_abstract.setObjectName(u"textBrowser_abstract")
        self.textBrowser_abstract.viewport().setProperty("cursor", QCursor(Qt.IBeamCursor))
        self.textBrowser_abstract.setReadOnly(False)

        self.gridLayout_2.addWidget(self.textBrowser_abstract, 4, 0, 1, 4)


        self.gridLayout.addWidget(self.groupBox, 2, 0, 1, 1)


        self.retranslateUi(Form)
        self.horizontalSlider_temperature.valueChanged.connect(self.spinBox_temperature.setValue)
        self.spinBox_temperature.valueChanged.connect(self.horizontalSlider_temperature.setValue)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Form", u"Title", None))
        self.pushButton_stop.setText(QCoreApplication.translate("Form", u"stop\n"
" reset", None))
        self.label_title.setText("")
        self.groupBox.setTitle(QCoreApplication.translate("Form", u"Abstract", None))
        self.label_temperature.setText(QCoreApplication.translate("Form", u"Temperature (Creativity)", None))
        self.label_number_titles.setText(QCoreApplication.translate("Form", u"#Titles", None))
        self.pushButton_enter.setText(QCoreApplication.translate("Form", u"Generate Title", None))
    # retranslateUi

