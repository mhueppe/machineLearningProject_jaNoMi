# author: Michael Hüppe
# date: 13.03.2025
# project: resources/evaluation/scorChatGpt.py

import openai
import pandas as pd
import json
import os
# Set up your OpenAI API key
openai.api_key = ""

def evaluate_multiple_models(data):
    # Define the prompt
    prompt = """
        You are an expert in academic writing and natural language processing. Your task is to evaluate generated **titles** based on their corresponding **paper abstracts** using multiple metrics. Each title should be rated on a scale from **1 to 10** (1 = poor, 10 = excellent) for the following categories:

        1. **Relevance** – Does the title accurately reflect the core content of the abstract?
        2. **Fluency & Readability** – Does the title read naturally and follow proper grammar?
        3. **Creativity** – Is the title engaging while still being accurate?
        4. **Coverage** – Does the title include all essential aspects of the abstract?

        The input is the abstract, the title and then generated titles from different models. Please give the scores for each model_name.
        ### **Response Format**

        Return the response **strictly** as a **valid JSON string**, following this format:

        json
        {
          "<model_name>": {
                "relevance": integer 1-10,
                "fluency_readability": integer 1-10,
                "creativity": integer 1-10,
                "coverage": integer 1-10,
              }
        }


        Ensure that the response is **fully valid JSON**, using **double quotes for keys/strings** and **no trailing commas**. **Do not include any introductory text or explanations outside the JSON output.**

    """
    prompt += f"Here is the data: {data}"
    # Send the prompt to the API
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant specializing in evaluating scientific paper titles."},
            {"role": "user", "content": prompt}

        ],
        response_format={ "type": "json_object"}
    )

    # Extract and return the result
    result = response['choices'][0]['message']['content'].strip()
    return result

if __name__ == '__main__':

    model_generated = pd.read_csv(r"data/arxiv_data_test_results_ultimate.csv")

    models = ['03_13_2025__09_49_19', '03_13_2025__09_48_48', '03_13_2025__09_48_03']
    model_scores = {
        m: {'relevance': [],
            'fluency_readability': [],
            'creativity': [],
            'coverage': []} for m in models
    }
    for row in range(5):
        try:
            data_row = model_generated[["abstract", "title"] + models].iloc[0].to_dict()
            evaluation_results = json.loads(evaluate_multiple_models(data_row))
            for m, r in evaluation_results.items():
                for metric, value in r.items():
                    model_scores[m][metric].append(value)
        except Exception as e:
            print(f"{e} for {row}")

    json.dump(model_scores, open(r"model_scores_ultimate.json", "w"), indent=4)
