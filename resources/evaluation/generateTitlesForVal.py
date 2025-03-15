# author: Michael Hüppe
# date: 25.01.2025
# project: resources/evaluation/generateTitlesForVal.py

import os

import h5py
import pandas as pd
import json
import os
import json
from resources.createModel import init_model
from resources.preprocessing.tokenizer import TokenizerBertHuggingFace
from resources.inference.generateSummary import GenerateSummary
from resources.training.transformer.transformer import Transformer


def process_models(model_names, data_path, output_path, n = 10):
    """
    Generates titles for each abstract in val_data.csv using a list of models and saves the updated dataset.

    Args:
        model_names (list): List of model names.
        data_path (str): Path to val_data.csv containing abstracts in the "abstract" column.
        vocab_path (str): Path to vocabulary JSON file for tokenizer.
        output_path (str): Path to save the updated dataset.
    """
    # Load dataset
    df = pd.read_csv(data_path, nrows=n)
    if os.path.isfile(output_path):
        df = pd.read_csv(output_path)
    if "abstract" not in df.columns:
        raise ValueError("The input CSV file must contain a column named 'abstract'.")

    # Iterate over each model
    for model_name in model_names:
        print(f"Processing model: {model_name}")
        model_path = fr"/informatik1/students/home/3hueppe/Documents/machineLearningProject_jaNoMi/trained_models/Transformer/{model_name}"
        if model_name in df.columns:
            print(f"Skip {model_name}")
            continue
        try:
            # Load model parameters
            train_params = json.load(open(os.path.join(model_path, "modelInfo.json")))
            model_params = train_params["model_parameters"]
            target_max_length = model_params["target_max_length"]
            context_max_length = model_params["context_max_length"]

            model_params["return_attention_scores"] = False
            model_params["bottle_neck"] = model_params.get("bottle_neck", 0)
            # model_params["feed_forward_cropped"] = model_params.get("feed_forward_cropped", False)

            # Initialize and load the model
            try:
                model = init_model(Transformer, model_params)
            except Exception:
                model_params["feed_forward_cropped"] = True
                model = init_model(Transformer, model_params)

            with h5py.File(os.path.join(model_path, "modelCheckpoint.weights.h5"), "r") as f:
                print(list(f.keys()))  # List of stored layers

            model.summary()
            model.load_weights(os.path.join(model_path, "modelCheckpoint.weights.h5"))

            # Initialize tokenizer and generator
            vocab_path = model_params["tokenizer_vocab_path"]
            tokenizer = TokenizerBertHuggingFace(vocab_path)
            titleGenerator = GenerateSummary(model, tokenizer, target_max_length, context_max_length)

            # Generate titles for each abstract
            generated_titles = []
            for i, abstract in enumerate(df["abstract"]):
                try:
                    titles = titleGenerator.summarize(abstract, beam_width=1, temperature=1.1,
                                                      return_attention_scores=False)
                    generated_titles.append(titles[0])
                except Exception as e:
                    print(f"Error generating title for abstract: {e}")
                    generated_titles.append("")

            # Add the generated titles as a new column
            df[model_name] = generated_titles
            print(f"Finished processing model: {model_name}")

        except Exception as e:
            print(f"Error loading or processing model {model_name}: {e}")

    # Save the updated dataset
    df.to_csv(output_path, index=False)
    print(f"Updated dataset saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Define model names
    #model_names = ["03_13_2025__15_14_32", "03_13_2025__15_15_10", "03_13_2025__15_15_52", "03_13_2025__15_16_52", "03_13_2025__15_17_23", "03_13_2025__15_18_08", "03_13_2025__15_18_50", "03_13_2025__15_19_56"] # vocab
    model_names = ["03_13_2025__09_49_19", "03_13_2025__09_48_48", "03_13_2025__09_48_03", "03_13_2025__09_46_47", "03_13_2025__09_44_27", "03_13_2025__09_42_24"] # ultimate

    # Paths
    data_path = r"/informatik1/students/home/3hueppe/Documents/machineLearningProject_jaNoMi/data/arxiv_data_test.csv"  # Path to the input CSV file
    output_path = data_path.replace(".csv", "_results_ultimate.csv")  # Output CSV file path

    # Process models and generate titles
    process_models(model_names, data_path, output_path, n=5000)
