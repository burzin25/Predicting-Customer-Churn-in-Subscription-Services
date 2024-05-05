
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def preprocessing(data):
    '''
    This function takes a pandas dataframe as input, performs preprocessing (like dropping rows with NaN values, etc.),
    and then returns a pandas dataframe.
    
    Args:
        data (pandas.DataFrame): The input DataFrame.
    
    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    '''
    print("Preprocessing...")
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    data = data.copy()
    
    # Drop rows with NaN values in any column
    data = data.dropna(how='any')
    print("Dropped rows with NaN values.")

    # Replace 'male' with 1 and 'female' with 2 in 'gender'
    gender_mapping = {'male': 1, 'female': 2}
    data['gender'] = data['gender'].map(gender_mapping)

    # Convert float date to datetime for date columns
    date_columns = ['registration_init_time', 'transaction_date', 'membership_expire_date', 'date']
    for col in date_columns:
        data[col] = pd.to_datetime(data[col], format='%Y%m%d', errors='ignore')
    print("Converted float date columns to datetime.")
    
    # Drop the 'msno' column
    #data = data.drop(columns=['msno'])

    # Reorder columns to move 'is_churn' towards the end
    if 'is_churn' in data.columns:
        churn_column = data.pop('is_churn')
        data['is_churn'] = churn_column

    print("Complete!")
    return data


def preprocessing_v2(data):
    '''
    Perform preprocessing on the provided DataFrame by dropping specified columns and rearranging the columns.

    Args:
        data (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    '''

    # Drop specified columns
    columns_to_drop = ["msno", "registration_init_time", "transaction_date", "membership_expire_date", "date"]
    data = data.drop(columns=columns_to_drop)

    # Rearrange columns to place "is_churn" at the end
    churn_column = data.pop("is_churn")
    data["is_churn"] = churn_column

    return data


def drop_outliers(df, threshold=1.5):
    """
    Drop rows containing outliers in each column of a DataFrame using the IQR method.

    Parameters:
    - df: DataFrame
        The DataFrame to drop outliers from.
    - threshold: float, optional (default=1.5)
        The threshold multiplier for determining outliers. A higher threshold will result in fewer outliers being detected.

    Returns:
    - df_cleaned: DataFrame
        A new DataFrame with rows containing outliers removed.
    """
    df_cleaned = df.copy()
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    return df_cleaned


def drop_rows_with_large_values(X, large_threshold=1e6):
    """
    Drop rows containing very large values from the feature matrix X.

    Parameters:
    - X: numpy.ndarray
        The feature matrix.
    - large_threshold: float, optional (default=1e6)
        The threshold for defining very large values.

    Returns:
    - X_cleaned: numpy.ndarray
        The feature matrix with rows containing very large values removed.
    """
    # Check for very large values
    large_rows = np.any(np.abs(X) > large_threshold, axis=1)
    if np.any(large_rows):
        print("Rows with very large values found in X. Dropping...")
        X_cleaned = X[~large_rows]  # Drop rows with large values
    else:
        X_cleaned = X.copy()  # If no large values found, return a copy of X
    return X_cleaned


def adjust_data_quantity(data: pd.DataFrame, churn_percent: float, data_size_percent: float = 100.0) -> pd.DataFrame:
    '''
    Adjust the quantity of not churned data points while keeping churned data points unchanged.

    Args:
        data (pandas.DataFrame): The input DataFrame.
        churn_percent (float): The percentage of churned data points in the output DataFrame.
        data_size_percent (float, optional): The percentage of data to be used for adjustment. Default is 100.0 (use all data).

    Returns:
        pandas.DataFrame: The adjusted DataFrame.
    '''
    print("Adjusting data quantity...")
    
    # Convert data_size_percent from percentage to fraction
    data_size = data_size_percent / 100.0
    
    # Apply data size adjustment
    data = data.sample(frac=data_size, random_state=0)
    
    # Calculate the number of churned and not churned data points
    churned_count = data['is_churn'].sum()
    not_churned_count = len(data) - churned_count
    
    # Calculate the desired number of not churned data points based on the percentage
    desired_not_churned_count = int(churned_count / (churn_percent / 100))
    
    # If the desired count is less than the current count, sample a subset of not churned data points
    if desired_not_churned_count < not_churned_count:
        # Sample a subset of not churned data points
        not_churned_data = data[data['is_churn'] == 0].sample(n=desired_not_churned_count, replace=False)
        churned_data = data[data['is_churn'] == 1]  # Keep churned data points unchanged
    else:
        # Repeat not churned data points to match the desired count
        repeat_factor = desired_not_churned_count // not_churned_count
        remainder = desired_not_churned_count % not_churned_count
        not_churned_data = pd.concat([data[data['is_churn'] == 0]] * repeat_factor)
        if remainder > 0:
            not_churned_data = pd.concat([not_churned_data, data[data['is_churn'] == 0].sample(n=remainder, replace=False)])
        churned_data = data[data['is_churn'] == 1]  # Keep churned data points unchanged
    
    # Concatenate churned and adjusted not churned data points
    adjusted_data = pd.concat([churned_data, not_churned_data])
    
    print("Data quantity adjustment complete!")
    
    # Calculate and print the ratio of churned to not churned data points
    ratio = churned_count / desired_not_churned_count
    print(f"Ratio of churned to not churned data points: {ratio:.2f}")
    
    # Print value counts after adjustment
    print("Value counts after adjustment:\n", adjusted_data['is_churn'].value_counts())
    
    # Plot count of each class
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    adjusted_data['is_churn'].value_counts().plot(kind='bar', color=['blue', 'orange'])
    plt.title('Count of Each Class')
    plt.xlabel('is_churn')
    plt.ylabel('Count')
    for i, value in enumerate(adjusted_data['is_churn'].value_counts()):
        plt.text(i, value, str(value), ha='center', va='bottom')
    
    # Plot ratio of class 1 to class 0
    plt.subplot(1, 2, 2)
    adjusted_data['is_churn'].value_counts(normalize=True).plot(kind='bar', color=['green', 'red'])
    plt.title('Ratio of Class 1 to Class 0')
    plt.xlabel('is_churn')
    plt.ylabel('Ratio')
    for i, value in enumerate(adjusted_data['is_churn'].value_counts(normalize=True)):
        plt.text(i, value, f"{value:.2f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
    
    return adjusted_data