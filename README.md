# Machine Learning Project WS24/25 - Group JaNoMi

This is a public repository to manage the Machine Learning Project of Jannis, Michael and Noah for WS 2024/25.

## Getting Started

To start the application, follow these steps:

1. **Create a new conda environment**:
   ```sh
   conda create --name myenv python=3.10
   conda activate myenv

2. **Install the necessary requirements**:
     ```sh
    pip install -r requirements.txt

3. **Then start the main.py to view the current gui application**: 
     ```sh
    python main.py

## Project Sketch: HEADLINER - A Scientific Title Generator
HEADLINER is a machine learning project aimed at developing a scientific title generator. The goal is to train a language model, specifically a transformer, from scratch using a dataset of abstract-title pairings. This model will then be capable of generating fitting titles for any given abstract. Additionally, it aims to be very data and resource efficient.
### Objectives

1. **Data Collection**: Gather a comprehensive dataset of scientific abstracts and their corresponding titles. 
    For now we will start out with the [ACL Title and Abstract Dataset](https://paperswithcode.com/dataset/acl-title-and-abstract-dataset).
2. **Model Training**: Train a transformer-based language model on the collected dataset.
3. **Title Generation**: Provide the user with an interface to input abstracts and generate appropriate titles using the trained model.
4. **Evaluation**: Implement evaluation metrics to assess the quality and relevance of the generated titles.

### Components

- **Data Analysis**: Scripts and notebooks for data preprocessing and analysis.
- **Model Training**: Code for training the transformer model.
- **GUI**: A graphical user interface for users to input abstracts and receive generated titles.
- **Evaluation**: Tools and scripts for evaluating the model's performance.

### Technologies

- **Python**: The primary programming language for the project.
- **TensorFlow**: We will be using TensorFlow for training the model.
- **PySide2**: For developing the GUI.
- **Scikit-learn**: For data preprocessing and evaluation metrics.
- **pytest**: For unit testing.
