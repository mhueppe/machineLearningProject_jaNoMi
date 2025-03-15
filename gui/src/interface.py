# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'interface.ui'
##
## Created by: Qt User Interface Compiler version 6.8.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractScrollArea, QApplication, QCheckBox, QComboBox,
    QGridLayout, QGroupBox, QLabel, QListWidget,
    QListWidgetItem, QProgressBar, QPushButton, QSizePolicy,
    QSlider, QSpacerItem, QSpinBox, QTextBrowser,
    QVBoxLayout, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(602, 750)
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setObjectName(u"gridLayout")
        self.groupBox_3 = QGroupBox(Form)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout_2 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.textBrowser_title = QTextBrowser(self.groupBox_3)
        self.textBrowser_title.setObjectName(u"textBrowser_title")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser_title.sizePolicy().hasHeightForWidth())
        self.textBrowser_title.setSizePolicy(sizePolicy)
        self.textBrowser_title.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        self.verticalLayout_2.addWidget(self.textBrowser_title)


        self.gridLayout.addWidget(self.groupBox_3, 2, 0, 1, 1)

        self.groupBox = QGroupBox(Form)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout_2 = QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")

        self.gridLayout_2.addWidget(self.label, 5, 3, 1, 1)

        self.comboBox_modelType = QComboBox(self.groupBox)
        self.comboBox_modelType.setObjectName(u"comboBox_modelType")

        self.gridLayout_2.addWidget(self.comboBox_modelType, 7, 2, 1, 2)

        self.spinBox_beam_width = QSpinBox(self.groupBox)
        self.spinBox_beam_width.setObjectName(u"spinBox_beam_width")
        self.spinBox_beam_width.setMinimum(1)
        self.spinBox_beam_width.setMaximum(10)
        self.spinBox_beam_width.setValue(3)

        self.gridLayout_2.addWidget(self.spinBox_beam_width, 6, 3, 1, 1)

        self.progressBar = QProgressBar(self.groupBox)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(False)

        self.gridLayout_2.addWidget(self.progressBar, 7, 0, 1, 1)

        self.horizontalSlider_temperature = QSlider(self.groupBox)
        self.horizontalSlider_temperature.setObjectName(u"horizontalSlider_temperature")
        self.horizontalSlider_temperature.setMinimum(1)
        self.horizontalSlider_temperature.setMaximum(100)
        self.horizontalSlider_temperature.setValue(30)
        self.horizontalSlider_temperature.setOrientation(Qt.Horizontal)

        self.gridLayout_2.addWidget(self.horizontalSlider_temperature, 6, 0, 1, 2)

        self.label_temperature = QLabel(self.groupBox)
        self.label_temperature.setObjectName(u"label_temperature")

        self.gridLayout_2.addWidget(self.label_temperature, 5, 0, 1, 1)

        self.spinBox_number_titles = QSpinBox(self.groupBox)
        self.spinBox_number_titles.setObjectName(u"spinBox_number_titles")
        self.spinBox_number_titles.setMinimum(1)
        self.spinBox_number_titles.setMaximum(20)
        self.spinBox_number_titles.setValue(5)

        self.gridLayout_2.addWidget(self.spinBox_number_titles, 6, 4, 1, 1)

        self.label_number_titles = QLabel(self.groupBox)
        self.label_number_titles.setObjectName(u"label_number_titles")

        self.gridLayout_2.addWidget(self.label_number_titles, 5, 4, 1, 1)

        self.spinBox_temperature = QSpinBox(self.groupBox)
        self.spinBox_temperature.setObjectName(u"spinBox_temperature")
        self.spinBox_temperature.setMinimum(1)
        self.spinBox_temperature.setMaximum(100)
        self.spinBox_temperature.setValue(30)

        self.gridLayout_2.addWidget(self.spinBox_temperature, 6, 2, 1, 1)

        self.checkBox_no_reps = QCheckBox(self.groupBox)
        self.checkBox_no_reps.setObjectName(u"checkBox_no_reps")

        self.gridLayout_2.addWidget(self.checkBox_no_reps, 6, 5, 1, 1)

        self.textBrowser_abstract = QTextBrowser(self.groupBox)
        self.textBrowser_abstract.setObjectName(u"textBrowser_abstract")
        self.textBrowser_abstract.viewport().setProperty(u"cursor", QCursor(Qt.CursorShape.IBeamCursor))
        self.textBrowser_abstract.setReadOnly(False)

        self.gridLayout_2.addWidget(self.textBrowser_abstract, 4, 0, 1, 6)

        self.pushButton_enter = QPushButton(self.groupBox)
        self.pushButton_enter.setObjectName(u"pushButton_enter")

        self.gridLayout_2.addWidget(self.pushButton_enter, 7, 4, 1, 2)


        self.gridLayout.addWidget(self.groupBox, 3, 0, 1, 1)

        self.groupBox_2 = QGroupBox(Form)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.gridLayout_3 = QGridLayout(self.groupBox_2)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.listWidget_titles = QListWidget(self.groupBox_2)
        self.listWidget_titles.setObjectName(u"listWidget_titles")
        self.listWidget_titles.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        self.gridLayout_3.addWidget(self.listWidget_titles, 0, 0, 1, 3)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer, 1, 0, 1, 1)

        self.pushButton_stop = QPushButton(self.groupBox_2)
        self.pushButton_stop.setObjectName(u"pushButton_stop")

        self.gridLayout_3.addWidget(self.pushButton_stop, 1, 1, 1, 1)


        self.gridLayout.addWidget(self.groupBox_2, 0, 0, 1, 1)


        self.retranslateUi(Form)
        self.horizontalSlider_temperature.valueChanged.connect(self.spinBox_temperature.setValue)
        self.spinBox_temperature.valueChanged.connect(self.horizontalSlider_temperature.setValue)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Form", u"Title", None))
#if QT_CONFIG(tooltip)
        self.textBrowser_title.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>Here the currently selected title is displayed. <br/>The sum of all cross attention heads in the last decoder layer are <br/>displayed to show the attention mechanism of the model.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.groupBox.setTitle(QCoreApplication.translate("Form", u"Abstract", None))
        self.label.setText(QCoreApplication.translate("Form", u"Beam Width", None))
        self.label_temperature.setText(QCoreApplication.translate("Form", u"Temperature (Creativity)", None))
        self.label_number_titles.setText(QCoreApplication.translate("Form", u"#Titles", None))
#if QT_CONFIG(tooltip)
        self.checkBox_no_reps.setToolTip(QCoreApplication.translate("Form", u"When this is clicked no repetition in the title are allowed ", None))
#endif // QT_CONFIG(tooltip)
        self.checkBox_no_reps.setText(QCoreApplication.translate("Form", u"No Reps", None))
#if QT_CONFIG(tooltip)
        self.textBrowser_abstract.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>Here you can input your abstract.<br/>After generating a title the sum of all cross attention heads is shown. <br/>You can see the different attention scores by selecting different titles in the output.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton_enter.setText(QCoreApplication.translate("Form", u"Generate Title", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Form", u"Output", None))
#if QT_CONFIG(tooltip)
        self.listWidget_titles.setToolTip(QCoreApplication.translate("Form", u"<html><head/><body><p>This is the output of the model. It's a live view of the beam search. </p><p>The top n beams are visible here.</p><p>Often the beam search finds the best options early one but still exploring the search space till the end. </p><p>This might result in the final attentions taking a while to display.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton_stop.setText(QCoreApplication.translate("Form", u"reset", None))
    # retranslateUi

