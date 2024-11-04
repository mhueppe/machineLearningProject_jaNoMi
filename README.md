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

# ML Project Sketch

Below are three initial ideas for potential projects:

### Idea One: Jeopardy Bot
- **Dataset**: 53MB JSON file containing 216,930 Jeopardy questions, answers, and additional information. See [source](http://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file/)
- **Goal**: Train a model that can play Jeopardy interactively, generating questions or answers as prompted by the user.
- **Architecture**:  The model will be a transformer-based language model trained from scratch.The dataset will be augmented with additional information about different topics to help the model generate questions and answers for new topics. Input features (x) include the topic area and possibly additional information, while the output (y) includes both the question and answer. The model will classify the type of question (e.g., history, politics, sports) and generate responses accordingly. A potential challenge is the risk of overfitting due to the structured nature of the data, so additional steps will need to be taken to reduce this risk.

### Idea Two: Amazon Review Generator
- **Dataset**: Either [Stanford's Amazon Dataset](https://snap.stanford.edu/data/web-Amazon.html) or [McAuley's Amazon Dataset](https://amazon-reviews-2023.github.io) could be used.
- **Goal**: Create a model that generates Amazon reviews for a product, based on a given product name and star rating. The generated reviews should align with the specified rating.
- **Architecture**:  The model will be a transformer-based language model trained from scratch.Input features (x) include the product name, relevant features, and star rating, while the output (y) is a coherent review matching the input criteria. The model will classify the review sentiment (e.g., bad, neutral, very good) based on the provided star rating and generate an appropriate response. Given the large amount of available data, the model is expected to generalize well, though there may be challenges in balancing creativity and realism. To emphasize the focus on data efficiency, we might need to constrain ourselves to a subset of those datasets.

### Idea Three: Pokedex Generator
- **Dataset**: Scraped data from an online Pokedex.
- **Goal**: Develop a model that can generate creative new entries for a Pokedex, describing fictional Pokémon characteristics in a natural and imaginative way.
- **Architecture**:  The model will be a transformer-based language model trained from scratch.The training process will involve first pre-training on Pokémon-relevant data such as attack descriptions, scripts, and dialogues to help the model understand the Pokémon universe, and then fine-tuning on the Pokedex dataset. Input features (x) include the Pokémon type, and the output (y) is a creative description of the Pokémon. The model will classify the type (e.g., water, fire, grass) and generate vivid, unique descriptions accordingly. The main challenge is the limited amount of available data, which may lead to underfitting, highlighting the importance of data efficiency in the training process.



