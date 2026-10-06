# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
# Load dataset
df = pd.read_csv( r'C:\Users\HP\Desktop\ML\Fake-news\dataset\FA-KES-Dataset.csv' ,encoding='latin1')
# Display first few rows of data
# print(df.head())
# Define features and targets
# Features
X = df['article_title'] + " " + df['article_content']
# Target
y = df['labels']
# Split dataset into training and test splits
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
# Initialize Tfidf Vectorizer ro convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Initialize naive bayes classifier
model = MultinomialNB()
# Train model
model.fit(X_train_tfidf,y_train)
# Make predictions on test set
y_pred = model.predict(X_test_tfidf)
# Evaluate model
accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)
print(f"\nAccuracy: {accuracy}")
print("\nClassification report:\n ",report)