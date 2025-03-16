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
from PySide6.QtWidgets import QWidget, QTextBrowser, QVBoxLayout, QListWidgetItem, QMessageBox
from resources.model_types import ModelTypes
from resources.preprocessing.dataPreprocessing import preprocessing
from resources.preprocessing.tokenizer import Tokenizer
from PySide6.QtWidgets import QApplication, QListWidget, QListWidgetItem, QLabel, QStyledItemDelegate
from PySide6.QtCore import Qt


class Worker(QtCore.QThread):
    output_ready = QtCore.Signal(list)

    def __init__(self, work_func: Callable,
                 user_input: str,
                 model_type: ModelTypes,
                 beam_width: int,
                 num_results: int,
                 temperature: float,
                 no_reps: bool,
                 gui_cb: Callable):
        super().__init__()
        self._work_func = work_func
        self._user_input = user_input
        self._model_type = model_type
        self._beam_width = beam_width
        self._num_results = num_results
        self._temperature = temperature
        self._no_reps = no_reps
        self._gui_cb = gui_cb

    def run(self):
        output = self._work_func(user_input=self._user_input,
                                 model_type=self._model_type,
                                 beam_width=self._beam_width,
                                 num_results=self._num_results,
                                 temperature=self._temperature,
                                 no_reps=self._no_reps,
                                 gui_cb=self._gui_cb)
        self.output_ready.emit(output)

    def stop(self):
        if self.isRunning():
            self.terminate()


class Interface(QWidget, Ui_Form):
    """
    Implementation of a simple GUI interface to input text and receive some kind of output.
    """
    output_send = QtCore.Signal(list)  # TODO: is this currently used?

    def __init__(self,
                 cb_inputEnter: Callable = lambda userInput, model: None,
                 cb_getTokenizer: Callable = lambda: None,
                 ):
        super().__init__()
        self._cb_inputEnter: Callable[[str, ModelTypes], None] = cb_inputEnter
        self._cb_getTokenizer: Callable[[ModelTypes], Tokenizer] = cb_getTokenizer
        self.setup_ui(self)
        self.setup_connections()
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
        self.textBrowser_abstract.setText(
            "The dominant sequence transduction models are based on complex recurrent or convolutional neural "
            "networks in an encoder-decoder configuration. The best performing models also connect the encoder and "
            "decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, "
            "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments "
            "on two machine translation tasks show these models to be superior in quality while being more "
            "parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT "
            "2014 English-to-German translation task, improving over the existing best results, including ensembles "
            "by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new "
            "single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, "
            "a small fraction of the training costs of the best models from the literature. We show that the "
            "Transformer generalizes well to other tasks by applying it successfully to English constituency parsing "
            "both with large and limited training data."
        )
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
        """
        Called when a different title has been selected
        :return:
        """
        selectedTitle_idx = self.listWidget_titles.currentRow()
        if selectedTitle_idx == -1 or selectedTitle_idx < len(self.output):
            self.displayAttention(self.output[selectedTitle_idx])

    def _pushButton_stop_clicked(self):
        """
        Reset all the views and stop the worker if necessary.
        :return:
        """
        try:
            self.worker.stop()
        except Exception:
            print('Worker cannot be stopped. probably because it doesnt exist')
        self.textBrowser_abstract.clear()
        self.listWidget_titles.clear()
        self.textBrowser_title.clear()

    def _pushButton_enter_clicked(self) -> None:
        """
        Send the current input to the input handler
        :return:
        """
        userInput = self.textBrowser_abstract.toPlainText()
        if len(userInput) == 0:
            # Create a message box
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)  # Set the icon to Warning
            msg_box.setWindowTitle("Warning")  # Set the title
            msg_box.setText("The abstract is empty. Please input one!")  # Set the main message
            msg_box.setStandardButtons(QMessageBox.Ok)  # Add an "OK" button
            # Show the message box
            msg_box.exec()
            return

        model_type = ModelTypes[self.comboBox_modelType.currentText()]
        self.listWidget_titles.clear()
        number_of_titles = self.spinBox_number_titles.value()
        beam_width = self.spinBox_beam_width.value()
        no_reps = self.checkBox_no_reps.isChecked()
        temperature = self.spinBox_temperature.value() / 100

        self.progressBar.setRange(0, 0)  # Indeterminate state (busy mode)
        self.listWidget_titles.addItems([""] * number_of_titles)
        self.worker = Worker(self._cb_inputEnter,
                             userInput,
                             model_type, beam_width, number_of_titles,
                             temperature, no_reps,
                             self.handle_stream)
        self.worker.output_ready.connect(self._on_worker_done)
        self.worker.start()

    def _on_worker_done(self, output: list):
        """
        Called when all the predictions have been made with the final
        best predicted sequences.
        :param output: Output containing the generated sequence as well as the attention values.
        :return:
        """
        self.output = output
        self.displayAttention(output[self.listWidget_titles.currentRow()])
        self.progressBar.setRange(0, 1)  # Indeterminate state (busy mode)

    def displayAttention(self, output):
        """
        Display the Attention for the title and the abstract
        :param output: Output containing the generated sequence as well as the attention values.
        :return:
        """
        prediction, score, attention = output
        # attention type, layer, head
        tokenizer = self._cb_getTokenizer(ModelTypes[self.comboBox_modelType.currentText()])
        tokens_y = "[START] " + tokenizer.detokenize(
            tokenizer.tokenize(preprocessing(prediction)))
        abstract = self.textBrowser_abstract.toPlainText()
        tokens_x = "[START] " + tokenizer.detokenize(
            tokenizer.tokenize(preprocessing(abstract)))
        try:
            x_length = len(tokens_x.split())
            html_content = generate_heatmap_text(tokens_x,
                                                 np.mean(np.mean(attention[2]["decoder_layer_1"][0], axis=0), axis=0)[
                                                 :x_length], "Greens", combine_tokens=True)
            self.textBrowser_abstract.setHtml(html_content)
            html_content = generate_heatmap_text(tokens_y,
                                                 np.mean(np.mean(attention[2]["decoder_layer_1"][0], axis=0), axis=1)[
                                                 :len(tokens_y.split())], "Greens", combine_tokens=True, title=True)
            self.textBrowser_title.setHtml(html_content)
        except Exception as e:
            print(e)
            pass

    def handle_stream(self, titles: list):
        """
        Append all the generated titles to the view
        :param titles: Titles to add to the view
        :return:
        """
        try:
            self.listWidget_titles.clear()
            self.listWidget_titles.addItems(titles)
        except Exception as e:
            print(e)
