# author: Michael Hüppe
# date: 25.01.2025
# project: resources/evaluation/generateTitlesForVal.py

import os
import pandas as pd
import json
import os
import json
from resources.createModel import init_model
from resources.preprocessing.tokenizer import TokenizerBertHuggingFace
from resources.inference.generateSummary import GenerateSummary
from resources.training.transformer.transformer import Transformer


def process_models(model_names, data_path, vocab_path, output_path):
    """
    Generates titles for each abstract in val_data.csv using a list of models and saves the updated dataset.

    Args:
        model_names (list): List of model names.
        data_path (str): Path to val_data.csv containing abstracts in the "abstract" column.
        vocab_path (str): Path to vocabulary JSON file for tokenizer.
        output_path (str): Path to save the updated dataset.
    """
    # Load dataset
    df = pd.read_csv(data_path, nrows=3000)
    if "abstract" not in df.columns:
        raise ValueError("The input CSV file must contain a column named 'abstract'.")

    # Iterate over each model
    for model_name in model_names:
        print(f"Processing model: {model_name}")
        model_path = fr"..\..\trained_models\Transformer\{model_name}"
        try:
            # Load model parameters
            train_params = json.load(open(os.path.join(model_path, "modelInfo.json")))
            model_params = train_params["model_parameters"]
            target_max_length = model_params["target_max_length"]
            context_max_length = model_params["context_max_length"]

            model_params["return_attention_scores"] = False
            model_params["bottle_neck"] = model_params.get("bottle_neck", 0)
            model_params["feed_forward_cropped"] = model_params.get("feed_forward_cropped", False)

            # Initialize and load the model
            try:
                model = init_model(Transformer, model_params)
            except Exception:
                model_params["feed_forward_cropped"] = True
                model = init_model(Transformer, model_params)

            model.summary()
            model.load_weights(os.path.join(model_path, "modelCheckpoint.weights.h5"))

            # Initialize tokenizer and generator
            tokenizer = TokenizerBertHuggingFace(vocab_path)
            titleGenerator = GenerateSummary(model, tokenizer, target_max_length, context_max_length)

            # Generate titles for each abstract
            generated_titles = []
            for abstract in df["abstract"]:
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
    model_names = [
        '01_25_2025__15_38_58', '01_25_2025__14_11_47', '01_25_2025__14_06_58', '01_25_2025__14_06_50',
        '01_25_2025__11_37_35', '01_25_2025__11_36_16', '01_25_2025__11_35_04', '01_25_2025__11_31_21',
        '01_25_2025__11_29_46', '01_25_2025__11_29_46', '01_24_2025__18_56_52', '01_24_2025__18_56_34',
        '01_24_2025__18_56_15', '01_24_2025__18_46_25', '01_24_2025__18_45_30', '01_24_2025__18_44_24',
        '01_24_2025__18_43_03', '01_24_2025__18_41_42', '01_24_2025__18_40_48', '01_24_2025__18_39_42',
        '01_24_2025__18_38_37', '01_24_2025__18_38_15', '01_23_2025__15_58_39', '01_23_2025__15_54_45',
        '01_23_2025__15_16_06', '01_23_2025__11_43_55', '01_23_2025__11_01_47', '01_23_2025__10_45_10',
        '01_21_2025__17_07_14', '01_21_2025__16_57_05', '01_21_2025__13_10_22', '01_20_2025__12_25_37',
        '01_20_2025__11_37_50', '01_20_2025__11_17_31', '01_20_2025__11_11_38', '01_15_2025__17_45_57',
        '01_15_2025__17_41_21', '01_15_2025__16_16_28', '01_15_2025__14_36_13', '01_15_2025__14_34_51',
        '01_15_2025__12_37_35', '01_15_2025__12_37_36', '01_14_2025__16_38_58', '01_14_2025__16_26_44',
        '01_14_2025__15_25_25', '01_14_2025__15_16_16', '01_13_2025__17_43_48', '01_13_2025__13_49_04',
        '01_08_2025__15_15_55', '01_08_2025__15_14_44', '01_08_2025__15_13_16', '01_08_2025__15_11_22',
        '01_08_2025__11_36_02', '01_08_2025__11_31_46', '01_08_2025__11_31_47', '01_08_2025__10_03_37',
        '01_08_2025__10_01_47', '01_08_2025__09_59_44', '01_08_2025__09_59_21', '01_07_2025__15_10_57',
        '01_06_2025__18_06_42', '01_06_2025__17_30_57', '01_06_2025__16_28_07', '01_06_2025__13_23_21',
        '01_06_2025__12_49_39', '01_06_2025__12_20_47', '01_06_2025__11_49_53', '12_21_2024__15_21_10',
        '12_21_2024__09_23_31', '12_21_2024__00_06_56', '12_17_2024__13_17_13', '12_16_2024__22_25_38',
        '12_16_2024__19_15_14'
    ]

    # Paths
    data_path = r"..\..\data\arxiv_data_val_keyWord_csv_section.csv"  # Path to the input CSV file
    vocab_path = r"..\..\arxiv_vocab_8000.json"  # Path to the vocabulary JSON
    output_path = data_path.replace(".csv", "_results.csv")  # Output CSV file path

    # Process models and generate titles
    process_models(model_names, data_path, vocab_path, output_path)
