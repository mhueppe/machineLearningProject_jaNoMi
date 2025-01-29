# author: Michael Hüppe
# date: 28.10.2024
# project: /interface.py
# built-in
from typing import Callable
import numpy as np
from PySide6 import QtCore

from resources.evaluation.attention_evaluation import generate_heatmap_text
# local
from .src.interface import Ui_Form

# external
from PySide6.QtWidgets import QWidget, QTextBrowser, QVBoxLayout, QListWidgetItem
from resources.model_types import ModelTypes
from resources.preprocessing.dataPreprocessing import preprocessing
from PySide6.QtWidgets import QApplication, QListWidget, QListWidgetItem, QLabel, QStyledItemDelegate
from PySide6.QtCore import Qt

class HtmlDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        text = index.data(Qt.DisplayRole)  # Get the item text
        label = QLabel()
        label.setTextFormat(Qt.RichText)  # Enable HTML rendering
        label.setText(text)
        label.setFixedWidth(option.rect.width())  # Ensure proper text wrapping
        label.render(painter, option.rect.topLeft())


class Worker(QtCore.QThread):
    output_ready = QtCore.Signal(list)

    def __init__(self, work_func: Callable,
                 user_input: str,
                 model_type: ModelTypes,
                 num_results: int,
                 temperature: float,
                 gui_cb: Callable):
        super().__init__()
        self._work_func = work_func
        self._user_input = user_input
        self._model_type = model_type
        self._num_results = num_results
        self._temperature = temperature
        self._gui_cb = gui_cb

    def run(self):
        print("worker started!")
        output = self._work_func(user_input=self._user_input,
                                 model_type=self._model_type,
                                 num_results=self._num_results,
                                 temperature=self._temperature,
                                 gui_cb=self._gui_cb)
        self.output_ready.emit(output)

    def stop(self):
        if self.isRunning():
            self.terminate()


# TODO: is this class needed or is there a better way?
class ClickableTextBrowser(QWidget):
    clicked = QtCore.Signal(str)  # Custom signal to emit text content when clicked

    def __init__(self, html_content):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Create a QTextBrowser for full HTML support
        self.text_browser = QTextBrowser()
        self.text_browser.setHtml(html_content)
        self.text_browser.setFixedHeight(50)  # Adjust height as needed
        self.text_browser.setStyleSheet("border: none;")  # Remove border
        self.text_browser.setOpenExternalLinks(False)  # Disable external links

        # Detect mouse press events on the QTextBrowser
        self.text_browser.viewport().installEventFilter(self)

        layout.addWidget(self.text_browser)
        self.setLayout(layout)

        self.html_content = html_content

    def eventFilter(self, source, event):
        if event.type() == event.MouseButtonPress:
            # Emit signal with the text content when clicked
            self.clicked.emit(self.html_content)
            # TODO: highlight selection
            # TODO: show related attention
            return True
        return super().eventFilter(source, event)


class Interface(QWidget, Ui_Form):
    """
    Implementation of a simple GUI interface to input text and receive some kind of output.
    """
    output_send = QtCore.Signal(list) # TODO: is this currently used?

    def __init__(self, cb_inputEnter: Callable = lambda userInput: None, tokenizer = None):
        super().__init__()
        self._cb_inputEnter: Callable[[str, ModelTypes], None] = cb_inputEnter
        self.setup_ui(self)
        self.setup_connections()
        self.tokenizer = tokenizer
        self.output = []

    def setup_ui(self, Form) -> None:
        """
        Setup the UI form containing all the UI elements
        :param Form: UI window
        :return:
        """
        # Initialize the form with the interface layout
        super().setupUi(Form)
        self.comboBox_modelType.addItems(ModelTypes._member_names_)
        # self.listWidget_titles.setItemDelegate(HtmlDelegate(self.listWidget_titles))  # Set the custom delegate

    def setup_connections(self) -> None:
        """
        Setup the connections between the gui elements and the associated logic
        :return:
        """
        self.pushButton_enter.clicked.connect(
            self._pushButton_enter_clicked
        )

        self.pushButton_stop.clicked.connect(
            self._pushButton_stop_clicked
        )

        self.listWidget_titles.currentItemChanged.connect(
            self._listWidget_titles_selectionChanged
        )

    def _listWidget_titles_selectionChanged(self):
        selectedTitle_idx = self.listWidget_titles.currentRow()
        if selectedTitle_idx == -1 or selectedTitle_idx < len(self.output):
            self.displayAbstractAttention(self.output[selectedTitle_idx])

    def _pushButton_stop_clicked(self):
        try:
            self.worker.stop() # TODO: implement a more peaceful way
        except Exception: print('Worker cannot be stopped. probably because it doesnt exist')
        self.textBrowser_abstract.clear()
        self.listWidget_titles.clear()


    def _pushButton_enter_clicked(self) -> None:
        """
        Send the current input to the input handler
        :return:
        """
        print("butten clicked!")
        model_type = ModelTypes[self.comboBox_modelType.currentText()]
        # TODO: kontrollelemente ausgrauen/ausblenden, wenn nicht Headliner asugewählt ist?
        #self.spinBox_temperature.setDisabled()
        self.progressBar.setValue(0)
        self.listWidget_titles.clear()
        if model_type == ModelTypes.Headliner:
            number_of_titles = self.spinBox_number_titles.value()
            temperature = self.spinBox_temperature.value() / 200
            self.listWidget_titles.addItems([""] * number_of_titles)
            #for _ in range(number_of_titles):
            #    widget_item = QListWidgetItem(self.listWidget_titles)  # Create a QListWidgetItem
            #    clickable_widget = ClickableTextBrowser("")  # Create the custom widget
            #    self.listWidget_titles.addItem(widget_item)  # Add the item to the list
            #    self.listWidget_titles.setItemWidget(widget_item, clickable_widget)  # Set the custom widget for the item

            # TODO: toPlainText könnte auch woanders hin
            self.worker = Worker(self._cb_inputEnter, self.textBrowser_abstract.toPlainText(), model_type, number_of_titles, temperature, self.handle_stream)
            self.worker.output_ready.connect(self._on_worker_done) # TODO: refactor?
            self.worker.start()
        else:
            self.listWidget_titles.addItem()
            self._cb_inputEnter(self.textEdit_abstract.toPlainText(), model_type)


    def _on_worker_done(self, output: list):
        print("Worker done!")
        self.output = output
        # for item in output:
        #     titles, scores, attention_scores = item
        # self.listWidget_titles.clear()
        # for o in output:
        #     prediction, score, attention = output

        self.displayAbstractAttention(output[self.listWidget_titles.currentRow()])

    def displayAbstractAttention(self, output):
        prediction, score, attention = output
        # attention type, layer, head
        tokens_y = "[START] " + self.tokenizer.detokenize(
            self.tokenizer.tokenize(preprocessing(prediction)))
        tokens_x = "[START] " + self.tokenizer.detokenize(
            self.tokenizer.tokenize(preprocessing(self.textBrowser_abstract.toPlainText())))
        html_content = generate_heatmap_text(tokens_x,
                                             np.mean(np.mean(attention[2]["decoder_layer_1"][0], axis=0), axis=0)[
                                             :len(tokens_x.split())], "Greens", combine_tokens=True)
        self.textBrowser_abstract.setHtml(html_content)

        # attention = '<span style="background-color: #79c67a; color: #000000; padding: 0 4px; border-radius: 4px;">attention</span> <span style="background-color: #62bb6d; color: #000000; padding: 0 4px; border-radius: 4px;">based</span> <span style="background-color: #00441b; color: #FFFFFF; padding: 0 4px; border-radius: 4px;">recurrent</span> <span style="background-color: #005020; color: #FFFFFF; padding: 0 4px; border-radius: 4px;">neural</span> <span style="background-color: #70c274; color: #000000; padding: 0 4px; border-radius: 4px;">network</span> <span style="background-color: #f7fcf5; color: #000000; padding: 0 4px; border-radius: 4px;">models</span>'

    def handle_exploratory_stream(self, row_id: int, title: str, progress: int):
        print(progress, row_id, title)
        # TODO: show as html so there will be a smooth transition
        item = self.listWidget_titles.item(row_id)
        if item:
            item.setText(title)
            #html_content = f"<p>{title}</p>"
            #item.text_browser.setHtml(html_content)
        else: print("handle_stream: no list item!")
        # TODO: progressBar currently crashes programm eventually
        #self.progressBar.setValue(progress)


    # TODO: wenn ein titel bereits angezeigt wird, dann nicht die position verändern?
    #  oder lieber wie bisher immer nach score sortieren lassen?
    def handle_stream(self, titles: list, progress: int):
        try:
            print(progress, titles[0])
            self.listWidget_titles.clear()
            self.listWidget_titles.addItems(titles)
        except Exception as e:
                print(e)
        # TODO: progressBar currently crashes programm eventually
        #self.progressBar.setValue(progress)
