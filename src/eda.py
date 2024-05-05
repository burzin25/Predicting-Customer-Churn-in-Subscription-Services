

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def count_plot(data, columns):
    """
    Plot count plots for specified columns using matplotlib in separate windows.
    
    Args:
    - data (pandas.DataFrame): The DataFrame containing the data.
    - columns (list): List of columns to plot.
    """
    print("Plotting count plots...")
    
    # Plot count plots for each column
    for col in columns:
        # Create a new figure for each plot with larger horizontal size
        plt.figure(figsize=(10, 5))
        
        # Create count plot
        sns.countplot(x=col, data=data, palette='Set2')
        plt.title(f'Count Plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
    
    # Show plots
    plt.show()
    print("Plots completed!")

    

def plot_is_churn(data):
    """
    Plot the distribution of the 'is_churn' column and check the ratio.

    Args:
    - data (pandas.DataFrame): The DataFrame containing the data.
    """
    # Count plot for 'is_churn'
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='is_churn', data=data, palette='Set2')
    plt.title('Distribution of is_churn')
    plt.xlabel('is_churn')
    plt.ylabel('Count')

    # Add count labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plt.show()

    # Calculate churn ratio
    churn_ratio = data['is_churn'].value_counts(normalize=True) * 100
    print("Churn ratio:")
    print(churn_ratio)

def correlation_plot(X, y):
    """
    Generate a heatmap to visualize the correlation between features (X) and target variable (y).

    Parameters:
    - X (DataFrame): Features DataFrame.
    - y (Series): Target variable Series.

    Returns:
    None

    This function concatenates the features DataFrame (X) and target variable Series (y) to create a combined DataFrame.
    It then calculates the correlation matrix between features and the target variable.
    Finally, it plots a heatmap to visualize the correlations, with annotations showing the correlation coefficients.
    The colormap 'coolwarm' is used to represent positive and negative correlations, and values are formatted to two decimal places.
    """
    
    # Concatenate X and y to create a DataFrame
    data = pd.concat([X, y], axis=1)
    
    # Calculate the correlation matrix
    corr_matrix = data.corr()

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap between X and y')
    plt.show()