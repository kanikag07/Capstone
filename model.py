import pickle
import pandas as pd
import numpy as np


class SentimentRecommender:
    """
    The SentimentRecommender class provides personalized product recommendations based on sentiment analysis of user reviews. 
    It utilizes pre-trained models and prepared data to generate recommendations tailored to a specific user's preferences.
    Key Attributes
    Model Paths: Paths to pre-trained models and data files, such as sentiment analysis model, TF-IDF vectorizer, recommendation engine, and cleaned data.
        best_sentiment_model.pkl
        tfidf.pkl
        best_recommendation_model.pkl
        clean_data.pkl
    Loaded Models and Data:
        Sentiment prediction model for classifying reviews as positive or negative.
        TF-IDF vectorizer for transforming textual reviews into numerical features.
        Recommendation engine for suggesting products.
        Cleaned product data with pre-processed reviews.
    Methods
    __init__:
        Initializes and loads all necessary models and data from their respective paths.
    top5_recommendations(user_name):
        Generates the top 5 product recommendations for a given user.
        Steps:
        Validates the existence of the user in the recommendation dataset.
        Fetches the top 20 products recommended for the user.
        Extracts reviews for these products and processes them using the TF-IDF vectorizer.
        Predicts the sentiment of each review using the sentiment analysis model.
        Aggregates sentiment data to calculate the percentage of positive reviews for each product.
        Sorts products by positive sentiment percentage and returns the top 5.

    """
    root_model_path = "models/"
    sentiment_model = "best_sentiment_model.pkl"
    tfidf_vectorizer = "tfidf.pkl"
    best_recommender = "best_recommendation_model.pkl"
    clean_dataframe = "clean_data.pkl"

    def __init__(self):
        self.sentiment_model = pickle.load(open(
            SentimentRecommender.root_model_path + SentimentRecommender.sentiment_model, 'rb'))
        self.tfidf_vectorizer = pd.read_pickle(
            SentimentRecommender.root_model_path + SentimentRecommender.tfidf_vectorizer)
        self.user_final_rating = pickle.load(open(
            SentimentRecommender.root_model_path + SentimentRecommender.best_recommender, 'rb'))
        self.cleaned_data = pickle.load(open(
            SentimentRecommender.root_model_path + SentimentRecommender.clean_dataframe, 'rb'))

    def top5_recommendations(self, user_name):
        if user_name not in self.user_final_rating.index:
            print(f"The User {user_name} does not exist. Please provide a valid user name")
            return None
        else:
            # Get top 20 recommended products from the best recommendation model
            top20_recommended_products = list(
                self.user_final_rating.loc[user_name].sort_values(ascending=False)[0:20].index)
            # Get only the recommended products from the prepared dataframe "df_sent"
            df_top20_products = self.cleaned_data[self.cleaned_data.id.isin(top20_recommended_products)]
            # For these 20 products, get their user reviews and pass them through TF-IDF vectorizer to convert the data into suitable format for modeling
            X = self.tfidf_vectorizer.transform(df_top20_products["reviews_lemmatized"].values.astype(str))
            # Use the best sentiment model to predict the sentiment for these user reviews
            df_top20_products['predicted_sentiment'] = self.sentiment_model.predict(X)
            # Create a new column to map Positive sentiment to 1 and Negative sentiment to 0. This will allow us to easily summarize the data
            df_top20_products['positive_sentiment'] = df_top20_products['predicted_sentiment'].apply(
                lambda x: 1 if x == "Positive" else 0)
            
            # Groupby product names and calculate sentiment counts
            pred_df = df_top20_products.groupby('name').agg(
                pos_sent_count=('positive_sentiment', 'sum'), 
                total_sent_count=('id', 'size')
            )

            # Debugging step: Check the shape and columns
            print(pred_df.shape)  # Check the number of rows and columns
            print(pred_df.columns)  # Check the column names

            # Handle missing or invalid values
            pred_df['pos_sent_count'] = pred_df['pos_sent_count'].fillna(0)
            pred_df['total_sent_count'] = pred_df['total_sent_count'].fillna(0)

            # Now calculate percentage of positive sentiment
            pred_df['post_sent_percentage'] = np.round(pred_df['pos_sent_count'] / pred_df['total_sent_count'] * 100, 2)

            # Return top 5 recommended products to the user
            result = list(pred_df.sort_values(by='post_sent_percentage', ascending=False)[:5].index)
            return result




