
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt


def feature_engineering(data):
    '''
    Perform feature engineering on the provided DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame containing the original features.

    Returns:
        pandas.DataFrame: The DataFrame with additional engineered features.
    '''

    # Convert relevant columns to datetime type
    data['registration_init_time'] = pd.to_datetime(data['registration_init_time'], format='%Y%m%d', errors='coerce')
    data['membership_expire_date'] = pd.to_datetime(data['membership_expire_date'], format='%Y%m%d', errors='coerce')
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d', errors='coerce')

    # Drop rows with NaT values
    data.dropna(subset=['registration_init_time', 'membership_expire_date', 'date'], inplace=True)

    # Extract year, month, and day from registration_init_time
    data['registration_year'] = data['registration_init_time'].dt.year
    data['registration_month'] = data['registration_init_time'].dt.month
    data['registration_day'] = data['registration_init_time'].dt.day

    # Calculate subscription duration
    data['subscription_duration'] = (data['membership_expire_date'] - data['registration_init_time']).dt.days

    # Extract month and day of the week from date
    data['month'] = data['date'].dt.month
    data['day_of_week'] = data['date'].dt.dayofweek

    # Calculate listening session frequency
    session_count = data.groupby('msno')['date'].count().reset_index()
    session_count.columns = ['msno', 'session_count']
    data = pd.merge(data, session_count, on='msno', how='left')

    # Calculate average listening time per session
    data['avg_listen_time_per_session'] = data['total_secs'] / data['session_count']

    # Calculate ratio of skipped songs
    data['skipped_ratio'] = (data['num_25'] + data['num_50']) / data['num_unq']

    return data




def select_best_features(X, y, k=10, score_func=f_classif):
    '''
    Selects the best features for a classification problem using SelectKBest.

    Args:
        X (pandas.DataFrame): The DataFrame containing the independent variables.
        y (pandas.Series): The Series containing the dependent variable.
        k (int): The number of top features to select. Default is 10.
        score_func (callable): The scoring function to use for feature selection. Default is f_classif.

    Returns:
        list: A list of selected feature names.
    '''
    # Initialize SelectKBest with the specified scoring function and k
    kb = SelectKBest(score_func=score_func, k=k)
    
    # Fit SelectKBest to the data
    kb.fit(X, y)
    
    # Get the scores and feature indices
    scores = kb.scores_
    indices = np.arange(len(scores))
    
    # Remove features with NaN scores
    non_nan_indices = indices[~np.isnan(scores)]
    non_nan_scores = scores[~np.isnan(scores)]

    # Get the indices of the selected features
    selected_indices = non_nan_indices[np.argsort(non_nan_scores)[::-1][:k]]

    # Get the names and scores of the selected features
    selected_features = [X.columns[i] for i in selected_indices]
    selected_scores = scores[selected_indices]
    
    # Print the statistical report of the selected features
    report = pd.DataFrame({'Feature': selected_features, 'Score': selected_scores})
    print(report.sort_values(by='Score', ascending=False))
    
    # Plot the feature importance
    plt.figure(figsize=(8, 6))
    plt.barh(report['Feature'], report['Score'], color='b')
    plt.xlabel('Score')
    plt.ylabel('Features')
    plt.title('Feature Importance')
    plt.show()
    
    return selected_features

