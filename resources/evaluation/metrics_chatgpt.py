import pandas as pd
import openai


generated_title = ""
abstract = ""


API_KEY = "key"
MODEL = "gpt-4o-mini"


def get_semantic_similarity_score(abstract, generated_title):
    prompt = f"""
    Evaluate the semantic similarity between the title and the abstract below.
        - Title: "{generated_title}"
        - Abstract: "{abstract}"
    
        On a scale of 1 to 10, where 1 means "completely unrelated" and 10 means "perfectly captures the main idea of the abstract," provide your score and explain your reasoning.
        Your response should be written in two different lines in the format: "score: X \n explanation: ..."
    """

    openai.api_key = API_KEY
    try:
        response = openai.chat.completions.create(
            messages=[{"role": "system", "content": "You are an expert in academic writing evaluation."},
                          {"role": "user", "content": prompt}],
                temperature=0.7,
            model=MODEL,
        )

        reply = response.choices[0].message.content
        lines = reply.splitlines()
        score = float(lines[0].split(":")[1].strip())
        explanation = lines[1].split(":")[1].strip() # may result in problems if the explanation contains ":" later on
        score, explanation = lines[:2]

        return score, explanation
    except Exception as e:
        print(e)
        return None, None


def get_extractive_to_abstractive_balance(abstract, generated_title):
    prompt = f"""
    Does the title '{generated_title}' directly copy phrases from the abstract '{abstract}', or is it a paraphrased summary? Rate this balance on a scale from 1 (fully extractive) to 10 (fully abstractive but still relevant).
    Your response should be written in two different lines in the format: "score: X \n explanation: ..."
    """
    try:
        response = openai.chat.completions.create(
            messages=[{"role": "system", "content": "You are an expert in academic writing evaluation."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            model=MODEL,
        )

        reply = response.choices[0].message.content
        lines = reply.splitlines()
        score = float(lines[0].split(":")[1].strip())
        explanation = lines[1].split(":")[1].strip()
        score, explanation = lines[:2]

        return score, explanation
    except Exception as e:
        print(e)
        return None, None


if __name__ == "__main__":
    #df = pd.read_csv('generated_titles.csv')
    df = pd.DataFrame(data=[["1","2","3"],["4","5","6"],["7","8","9"]], columns=["abstract","title"])
    results = []
    for abstract, generated_title in df["abstract","title"]:
        semantic_score, semantic_explanation = get_semantic_similarity_score(abstract, generated_title)
        extractive_score, extractive_explanation = get_extractive_to_abstractive_balance(abstract, generated_title)

        results.append({
            "semantic_score": semantic_score,
            "semantic_explanation": semantic_explanation,
            "extractive_score": extractive_score,
            "extractive_explanation": extractive_explanation,
        })

    # Create a new DataFrame from the results and merge with the original
    results_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    df.to_csv('generated_scores.csv')