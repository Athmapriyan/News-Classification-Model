import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  
from sklearn.naive_bayes import MultinomialNB 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re

data = pd.read_csv(r"D:\sem3\debug\BBC News Train.csv") 

print(data.columns)
article_column = 'Text'
category_column = 'Category' 
title_mapping = {title: idx for idx, title in enumerate(data[category_column].unique())}
data['title_numerical'] = data[category_column].map(title_mapping)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

def preprocess(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text
    return ""
train_data['processed_text'] = train_data[article_column].apply(preprocess)
test_data['processed_text'] = test_data[article_column].apply(preprocess)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_data['processed_text'])
X_test = vectorizer.transform(test_data['processed_text'])
y_train = train_data['title_numerical']
y_test = test_data['title_numerical']
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"Validation Accuracy: {accuracy:.16f}")
print(f"Precision: {precision:.16f}")
print(f"Recall: {recall:.16f}")
print(f"F1-score: {f1:.16f}")
