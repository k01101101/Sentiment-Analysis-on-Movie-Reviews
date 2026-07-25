# Sentiment Analysis on IMDb Movie Reviews

## Project Overview

This project is a Natural Language Processing (NLP) application that
classifies IMDb movie reviews according to their sentiment. Using text
preprocessing, TF-IDF feature extraction, and a Logistic Regression
model, the system predicts whether a review expresses a **Positive** or
**Negative** opinion. A rule-based **Neutral** category is also added
using TextBlob polarity scoring, so reviews can fall into three classes:
Positive, Negative, or Neutral. Two Word Clouds (one for positive
reviews, one for negative) are generated to visualize the most
frequently occurring words in each group.

## Objectives

-   Preprocess textual movie reviews.
-   Convert text into numerical features using TF-IDF.
-   Train a Logistic Regression sentiment classifier (Positive/Negative).
-   Add a Neutral category using TextBlob polarity scoring.
-   Evaluate the model using standard metrics.
-   Generate Word Clouds split by sentiment.
-   Save the trained model and vectorizer for future predictions.
-   Summarize findings as business insights.

## Tools & Technologies

-   Python
-   Jupyter Notebook
-   Pandas
-   NLTK
-   Scikit-learn
-   Matplotlib
-   WordCloud
-   TextBlob
-   Joblib

## Project Structure

    Sentiment_Analysis_Project/
    │── Sentiment_Analysis_IMDb.ipynb
    │── movie.csv.gz
    │── sentiment_model.pkl
    │── wordcloud.png
    │── README.md
    │── Project_Report.docx

## Dataset

The project uses the IMDb Movie Reviews dataset (`movie.csv.gz`, 40,000
reviews) containing `text` and `label` columns. The dataset is
well balanced (roughly 50/50 Positive/Negative) and contains a small
number of duplicate rows, which are removed during preprocessing.

## Execution Steps

### 1. Install Dependencies

``` bash
pip install pandas nltk scikit-learn matplotlib wordcloud joblib textblob
```

### 2. Place the Dataset

Place `movie.csv.gz` in the same directory as the notebook. `pandas`
reads `.gz` files directly, so no manual extraction is required.

### 3. Open the Notebook

Open `Sentiment_Analysis_IMDb.ipynb` using Jupyter Notebook or VS Code.

### 4. Run All Cells

Run the notebook from top to bottom. The notebook will: - Load the
dataset and remove duplicates/empty rows. - Clean and preprocess the
text. - Add a Neutral label using TextBlob polarity scoring. -
Generate TF-IDF features. - Train the Logistic Regression model
(stratified train/test split). - Evaluate the model. - Generate
Positive and Negative Word Clouds. - Save the trained model and
vectorizer as `sentiment_model.pkl`.

### 5. Generated Files

After execution, the following files will be available: -
`sentiment_model.pkl` - `wordcloud.png`

## Performance Evaluation

The model is evaluated using: - Accuracy - Precision - Recall -
F1-Score - Confusion Matrix

## Business Insights

-   **Fast feedback at scale**: instead of manually reading thousands
    of reviews, the model sorts them into Positive/Negative/Neutral
    in seconds.
-   **Word clouds reveal *why***: the Positive and Negative word
    clouds highlight the specific words driving good or bad reviews,
    pointing to what to keep doing or fix.
-   **Confusion matrix shows where the model struggles**: misclassified
    reviews are often the borderline/Neutral ones, which is useful to
    know when deciding how much to trust automated scoring.
-   **Neutral reviews carry signal too**: they often hold calmer,
    detailed feedback that gets missed when only extreme opinions are
    analyzed.
-   **Reusable across domains**: the same pipeline (cleaning, TF-IDF,
    Logistic Regression) can be applied to product reviews, customer
    support tickets, or social media comments with minimal changes.

## Future Enhancements

-   Replace the rule-based Neutral label with a properly labeled
    3-class training dataset (instead of TextBlob polarity scoring)
-   BERT or LSTM models
-   Real-time web application
-   Multilingual sentiment analysis

## Author

Submitted as part of an academic NLP project.
