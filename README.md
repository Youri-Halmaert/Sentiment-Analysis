# Sentiment Analysis of Tweets Using Traditional and Deep Learning Approaches

## Overview
This project compares the performance of traditional machine learning models (Logistic Regression and Naive Bayes) with a deep learning model (BERT) for sentiment analysis on Twitter data. The goal is to classify tweets into three sentiment categories: positive, neutral, or negative. The results demonstrate that BERT significantly outperforms traditional approaches, highlighting the advantages of contextualized language representations in sentiment analysis.

## Key Features
- **Data Preprocessing**: Includes text normalization, removal of URLs, mentions, hashtags, and punctuation, stopword removal, and lemmatization.
- **Traditional Models**: Logistic Regression and Naive Bayes with TF-IDF vectorization.
- **Deep Learning Model**: Fine-tuned BERT (`bert-base-uncased`) for contextual understanding.
- **Evaluation Metrics**: Accuracy and macro-averaged F1-score for balanced performance assessment.
- **Interpretability**: Analysis of influential features for traditional models.

## Results

| Model                | Accuracy | Macro F1-score |
|---------------------|----------|----------------|
| **BERT**            | 76.0%    | 0.76           |
| Logistic Regression | 69.0%    | 0.69           |
| Naive Bayes         | 64.0%    | 0.63           |

BERT showed superior performance, particularly in handling ambiguous and context-dependent language.

## Installation

### Clone the repository
```bash
git clone https://github.com/yourusername/sentiment-analysis-tweets.git
cd sentiment-analysis-tweets
