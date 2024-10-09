
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv('D:/sem3/debug/bbc_news.csv')
print(data.columns)
article_column = 'description'  
category_column = 'pubDate'  
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer
import re
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text
train_data['processed_text'] = train_data[article_column].apply(preprocess)
test_data['processed_text'] = test_data[article_column].apply(preprocess)
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data['processed_text'])
X_test = vectorizer.transform(test_data['processed_text'])
y_train = train_data[category_column]
y_test = test_data[category_column]
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print(classification_report(y_test, y_pred))
