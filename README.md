# Headliner

Headliner is a GUI-based application that generates titles for scientific papers based on their abstracts using pre-trained language models. The application supports different model architectures and allows users to experiment with various decoding parameters to influence title generation.

## Installation and Setup

To run the application, install the required dependencies:

```bash
pip install -r requirements.txt
```

Then, start the GUI using:

```bash
python main.py
```

## Directory Structure

- `gui/` - Contains the GUI code for the application.
- `resources/training/` - Code for model training.
- `resources/dataAnalysis/` - Code for data analysis.
- `resources/evaluation/` - Code for model evaluation.
- `resources/inference/` - Code for inference and prediction.
- `resources/preprocessing/` - Code for preprocessing data.
- `resources/utils/` - Utility functions for the application.
- `data/` - Directory for datasets. If using custom models, place the arXiv dataset here.

## Using Custom Models

To train and use a custom model, add the arXiv dataset to the `data/` subdirectory and modify the training scripts accordingly.

## GUI Components

### Output

- Displays the live view of the beam search.
- Shows the top `n` beams.
- Often, the best titles are found early, but the model continues exploring the search space.
- Final attention visualizations may take some time to display.

### Title

- Displays the currently selected generated title.
- Shows the sum of all cross-attention heads in the last decoder layer to illustrate how the model attends to different parts of the input.

### Abstract

- Input field where users can enter their abstract.
- After title generation, it displays the sum of all cross-attention heads.
- Users can analyze attention scores by selecting different titles in the output section.

### Decoding Parameters

- **Beam Width**: Defines how many candidate sequences the model considers at each step.
- **Temperature (Creativity)**: Controls randomness in title generation. Higher values lead to more diverse results.
- **No Reps Toggle**: Prevents duplicate words in generated sequences.
- **Model Selection**: Choose between three pre-trained models:
  - **Decoder-only**
  - **Medi**
  - **Maxi**

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

For any questions or contributions, feel free to open an issue or submit a pull request!

