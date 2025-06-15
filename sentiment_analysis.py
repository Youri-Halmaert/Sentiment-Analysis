import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
warnings.filterwarnings('ignore')

class TweetSentimentAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions (@user)
        text = re.sub(r'@[\w_]+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#[\w_]+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                 if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)

    def load_data(self):
        self.train_df = pd.read_csv('train.csv')
        self.test_df = pd.read_csv('test.csv')
        
        # Display class distribution
        print("\nClass Distribution:")
        print(self.train_df['sentiment'].value_counts(normalize=True))
        

        
        # Preprocess text
        self.train_df['clean_text'] = self.train_df['text'].apply(self.preprocess_text)
        self.test_df['clean_text'] = self.test_df['text'].apply(self.preprocess_text)



    def train_models(self):
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            self.train_df['clean_text'], 
            self.train_df['sentiment'],
            test_size=0.2,
            random_state=42,
            stratify=self.train_df['sentiment']
        )

        # Convert labels to numerical values
        label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        y_train_num = y_train.map(label_map)
        y_val_num = y_val.map(label_map)

        # Initialize BERT model and tokenizer
        model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label={0: 'positive', 1: 'neutral', 2: 'negative'},
            label2id=label_map
        )

        # Define custom dataset class
        class SentimentDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = self.labels[idx]
                return item

            def __len__(self):
                return len(self.labels)

        # Prepare datasets for BERT
        def tokenize_data(texts, labels):
            encodings = self.tokenizer(texts.tolist(), 
                                    truncation=True, 
                                    padding=True, 
                                    max_length=128,
                                    return_tensors='pt')
            return encodings, torch.tensor(labels.tolist())

        # Create datasets
        train_encodings, train_labels = tokenize_data(X_train, y_train_num)
        val_encodings, val_labels = tokenize_data(X_val, y_val_num)

        train_dataset = SentimentDataset(train_encodings, train_labels)
        val_dataset = SentimentDataset(val_encodings, val_labels)

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10
        )

        # Define metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            f1 = f1_score(labels, predictions, average='macro')
            return {'f1': f1}

        # Create data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding='longest'
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.bert_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator
        )

        # Train BERT model
        print("\nTraining BERT model...")
        trainer.train()

        # Save the best model
        self.bert_model.save_pretrained('./best_bert_model')
        self.tokenizer.save_pretrained('./best_bert_model')

        # Evaluate BERT model
        print("\nBERT Model Results:")
        predictions = trainer.predict(val_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        print("Classification Report:")
        print(classification_report(y_val_num, pred_labels))
        
        # Get the best model
        self.best_model = self.bert_model
        
        # Add BERT to the models dictionary
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'BERT': self.bert_model
        }

        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_val_tfidf = self.vectorizer.transform(X_val)
        X_test_tfidf = self.vectorizer.transform(self.test_df['clean_text'])

        # Train models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': MultinomialNB()
        }

        self.best_model = None
        best_f1 = 0

        for name, model in models.items():
            model.fit(X_train_tfidf, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_val_tfidf)
            f1 = f1_score(y_val, y_pred, average='macro')
            
            if f1 > best_f1:
                best_f1 = f1
                self.best_model = model
                
            print(f"\n{name} Results:")
            print("Classification Report:")
            print(classification_report(y_val, y_pred))
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_val, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['positive', 'neutral', 'negative'],
                       yticklabels=['positive', 'neutral', 'negative'])
            plt.title(f'Confusion Matrix - {name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
            plt.close()

    def generate_predictions(self):
        # Create label map
        label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        
        # Function to predict using BERT
        def predict_bert(texts):
            inputs = self.tokenizer(
                texts.tolist(),
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1).numpy()
                
            # Convert numerical labels back to text
            inv_label_map = {v: k for k, v in label_map.items()}
            return [inv_label_map[p] for p in predictions]

        # Generate predictions based on the best model
        if isinstance(self.best_model, torch.nn.Module):  # If BERT model
            test_predictions = predict_bert(self.test_df['clean_text'])
        else:  # If traditional model
            test_predictions = self.best_model.predict(
                self.vectorizer.transform(self.test_df['clean_text'])
            )
        
        # Create submission file
        submission_df = pd.DataFrame({
            'text': self.test_df['text'],
            'predicted_sentiment': test_predictions
        })
        
        submission_df.to_csv('sentiment_predictions.csv', index=False)
        print("\nPredictions saved to sentiment_predictions.csv")

    def interpret_model(self):
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get coefficients for logistic regression
        if isinstance(self.best_model, LogisticRegression):
            coef = self.best_model.coef_
            
            # Create DataFrame for coefficients
            coef_df = pd.DataFrame(coef, columns=feature_names, 
                                index=['positive', 'neutral', 'negative'])
            
            # Get top 10 features for each sentiment based on their actual contribution
            top_features = {}
            
            # For each sentiment, find words that most strongly predict that sentiment
            for sentiment in ['positive', 'neutral', 'negative']:
                # Get coefficients for this sentiment
                sentiment_coefs = coef_df.loc[sentiment]
                # Sort by absolute value and take top 10
                sentiment_coefs = sentiment_coefs.abs().nlargest(10)
                # Get the actual words and their coefficients
                top_features[sentiment] = [(word, coef_df.loc[sentiment, word]) 
                                          for word in sentiment_coefs.index]
            
            print("\nTop Features for Each Sentiment:")
            for sentiment, features in top_features.items():
                print(f"\n{sentiment.title()}")
                for word, coef in features:
                    # Explain the relationship between coefficient and sentiment
                    if coef > 0:
                        effect = "decreases probability of being"
                        sentiment_effect = "negative"
                    else:
                        effect = "increases probability of being"
                        sentiment_effect = "positive"
                    print(f"  {word}: {coef:.4f} ({effect} {sentiment} sentiment)")
            
            # Print examples of how words contribute to different sentiments
            print("\nExamples of how words contribute to different sentiments:")
            for word in ['love', 'hate', 'happy', 'sad', 'good', 'bad']:
                if word in feature_names:
                    print(f"\nWord: {word}")
                    print("Sentiment contributions:")
                    for sentiment in ['positive', 'neutral', 'negative']:
                        coef = coef_df.loc[sentiment, word]
                        # Explain the relationship between coefficient and sentiment
                        if coef > 0:
                            effect = "decreases probability of being"
                            sentiment_effect = "negative"
                        else:
                            effect = "increases probability of being"
                            sentiment_effect = "positive"
                        print(f"  {sentiment}: {coef:.4f} ({effect} {sentiment} sentiment)")
                
            # Print examples of how words contribute to different sentiments
            print("\nExamples of how words contribute to different sentiments:")
            for word in ['love', 'hate', 'happy', 'sad', 'good', 'bad']:
                if word in feature_names:
                    print(f"\nWord: {word}")
                    print("Sentiment contributions:")
                    for sentiment in ['positive', 'neutral', 'negative']:
                        coef = coef_df.loc[sentiment, word]
                        print(f"  {sentiment}: {coef:.4f}")

if __name__ == "__main__":
    analyzer = TweetSentimentAnalyzer()
    analyzer.load_data()
    analyzer.train_models()
    analyzer.generate_predictions()
    analyzer.interpret_model()
