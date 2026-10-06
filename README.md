# 📰 Fake News Detection Using Machine Learning

## 📌 Project Description

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to classify news articles as **fake or real**. The model is trained on the **FA-KES (Fake News around the Syrian War)** dataset and uses both the article title and article content as input features.

The text data is transformed into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**, after which a **Multinomial Naive Bayes** classifier is trained to predict the news classification.

This project demonstrates how machine learning can be applied to text classification and misinformation detection.

---

## 🎯 Objectives

- Detect whether a news article is fake or real.
- Apply Natural Language Processing to news articles.
- Convert text into numerical features using TF-IDF.
- Train a Multinomial Naive Bayes classification model.
- Evaluate the model using accuracy and classification metrics.

---

## 📊 Dataset

The project uses the **FA-KES (Fake News around the Syrian War)** dataset.

The dataset contains information such as:

- `unit_id` — Unique article identifier
- `article_title` — News article title
- `article_content` — Full article content
- `source` — News source
- `date` — Publication date
- `location` — Location associated with the article
- `labels` — Classification label

The model uses:

```text
article_title + article_content
```

as the input features and:

```text
labels
```

as the target variable.

---

## 🧠 Machine Learning Workflow

```text
FA-KES Dataset
      ↓
Data Loading
      ↓
Feature Selection
      ↓
Combine Article Title + Content
      ↓
Train/Test Split
      ↓
TF-IDF Vectorization
      ↓
Multinomial Naive Bayes
      ↓
Predictions
      ↓
Model Evaluation
```

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **Scikit-learn**
- **TF-IDF**
- **Multinomial Naive Bayes**
- **Natural Language Processing (NLP)**

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Vedline2547/Fake-news.git
```

Navigate to the project directory:

```bash
cd Fake-news
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required libraries:

```bash
pip install pandas scikit-learn
```

---

## 🚀 How to Run

Place the dataset inside the `dataset` folder and run:

```bash
python main.py
```

The program will train the model and display the classification accuracy and classification report.

---

## 📈 Model Evaluation

The model is evaluated using:

### Accuracy

Measures the percentage of correctly classified news articles.

### Classification Report

The classification report provides:

- Precision
- Recall
- F1-score
- Support

Example output:

```text
Accuracy: 0.XX

Classification report:

              precision    recall    f1-score    support

           0       0.XX      0.XX       0.XX        XX
           1       0.XX      0.XX       0.XX        XX

    accuracy                           0.XX        XX
   macro avg       0.XX      0.XX       0.XX        XX
weighted avg       0.XX      0.XX       0.XX        XX
```

The actual results may vary depending on the dataset and model configuration.

---

## 🔍 Feature Extraction

Since machine learning models cannot directly process raw text, the project uses **TF-IDF Vectorization**.

```python
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7
)
```

TF-IDF assigns numerical importance to words based on how frequently they appear in an article and how common they are across the dataset.

---

## 🤖 Machine Learning Model

The project uses **Multinomial Naive Bayes**:

```python
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
```

Multinomial Naive Bayes is a commonly used algorithm for text classification because it works well with high-dimensional word-frequency features such as TF-IDF.

---

## 📁 Project Structure

```text
Fake-news/
│
├── dataset/
│   └── FA-KES-Dataset.csv
│
├── main.py
│
└── README.md
```

> **Note:** Large datasets generally should not be committed directly to GitHub if they exceed repository or licensing limits. Check the dataset's usage and redistribution terms before uploading it.

---

## 🔮 Future Improvements

Possible improvements include:

- Comparing Naive Bayes with Logistic Regression and SVM.
- Performing text preprocessing and cleaning.
- Hyperparameter tuning.
- Handling class imbalance.
- Using n-grams with TF-IDF.
- Testing word embeddings.
- Experimenting with deep learning models such as LSTM.
- Developing a web interface using Flask or Streamlit.
- Deploying the trained model as a web application.

---

## 📚 Skills Demonstrated

- Python programming
- Data preprocessing
- Natural Language Processing
- Text classification
- Feature engineering
- TF-IDF vectorization
- Machine learning
- Model evaluation
- Scikit-learn

---

## 👨‍💻 Author

**Vedline Ochieng**

Civil Engineering Student | ML & AI Enthusiast | Python Developer

---

## ⭐ Project

If you find this project useful, feel free to star the repository and explore the code.
