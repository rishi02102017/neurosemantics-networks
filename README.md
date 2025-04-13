# 🔍 Deep Sentiment Analysis with FFNN & LSTM (PyTorch + spaCy)

<p align="center">
  <img src="SA.jpg" alt="SA" width="500"/>
</p>


A robust and extensible sentiment classification pipeline using **Feed-Forward Neural Networks (FFNN)** and **Recurrent Neural Networks (LSTM)** for both **binary** and **multi-class sentiment analysis**. This project leverages `PyTorch`, `spaCy`, and `scikit-learn` and evaluates performance on real-world datasets: **IMDB**, **Sentiment140**, and **Twitter**.

---

##  Features

-  Tokenization with `spaCy` + custom vocabulary builder
-  FFNN with averaged embeddings and dual hidden layers
-  LSTM-based classifier with final hidden state for classification
-  Supports both binary (IMDB, Sentiment140) and multi-class (Twitter) tasks
-  Logs train/dev metrics: accuracy, loss, precision, recall, F1
-  Visualizations for training curves (Loss, Accuracy, F1)
-  Sample inference with decoded input sentences

---

##  Tech Stack

| Component | Technology |
|----------|------------|
| Language | Python 3.x |
| Framework | PyTorch |
| Tokenizer | spaCy (`en_core_web_sm`) |
| Metrics | scikit-learn |
| Plotting | Matplotlib |
| DataLoaders | PyTorch native |
| Optional | CUDA/GPU support |

---

##  Datasets

| Dataset | Type | Labels |
|--------|------|--------|
| [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) | Binary | Positive / Negative |
| [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) | Binary | Positive / Negative |
| [Twitter Dataset](https://github.com/datasets/sentiment-analysis/blob/master/twitter_training.csv) | Multi-class | Positive / Negative / Neutral |

---

##  Installation

```bash
# Clone the repo
git clone https://github.com/your-username/deep-sentiment-nlu.git
cd deep-sentiment-nlu

# Install required packages
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

---

##  How to Run

Run the full multi-dataset sentiment classification pipeline using:

```bash
jupyter notebook Codebook.ipynb
```

Inside the notebook, you can toggle between IMDB, Sentiment140, and Twitter sections using provided cell headers and execute the full training + evaluation + inference workflows.

---

##  Architecture Overview

### 🔹 Feed-Forward Neural Network (FFNN)
```
Input → Embedding Layer → Avg Pooling → Linear(300→256) → ReLU → Linear(256→128) → ReLU → Linear(128→Output)
```

### 🔹 LSTM Network
```
Input → Embedding → LSTM(300→256) → Final Hidden State → Linear(256→Output)
```

> Output layer:  
> - Binary → 1 unit with BCEWithLogitsLoss  
> - Multi-Class → N units with CrossEntropyLoss

---

##  Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Plots: Loss vs Epochs, Accuracy vs Epochs, F1 Score

All results are saved/logged after each epoch for train/dev.

---

##  Sample Output

```
Input Sentence: this movie was absolutely wonderful , i loved every part
True Label: Positive
Predicted Label: Positive
Probability: 0.9672
```

---

##  Project Structure

```
.
├── Codebook.ipynb              # Main notebook with all training/evaluation
├── data/                       # Preprocessed or raw dataset files
├── plots/                      # Training visualizations
├── README.md
└── requirements.txt
```

---

##  Future Improvements

- Add attention mechanism to LSTM
- Use pre-trained embeddings (GloVe/FastText)
- Enable Transformer-based comparison (BERT)
- Improve handling of emoji/emoticons in tweets

---

## 👨‍💻 Authors

- Jyotishman Das ([@rishi02102017](https://github.com/rishi02102017))
- Denzel Lenshanglen Lupheng
- Suvadip Chakraborty
- Mehul Sah

---

##  License

MIT License

---

## ⭐️ Show your support

If you found this project useful or inspiring:
- Leave a ⭐️ on GitHub
- Fork it and build your own sentiment tools!

---
