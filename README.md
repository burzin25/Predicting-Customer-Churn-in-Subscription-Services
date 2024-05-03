
# Project Title

Predicting Customer Churn in Subscription Services

# Introduction
This project focuses on churn prediction in subscription-based businesses, where retaining customers is vital for sustainability. We aim to leverage data analytics and machine learning to develop predictive models that identify factors influencing churn. By analyzing historical user data, our goal is to empower businesses with actionable insights to proactively retain customers and enhance long-term success.

# Why customer churn prediction is important and how it can benefit businesses.

- **Prevents revenue loss**: By identifying customers at risk of churning, businesses can implement strategies to retain them, thereby preserving revenue streams.
- **Enhances customer satisfaction**: Proactive intervention to prevent churn demonstrates attentiveness to customer needs, fostering satisfaction and loyalty.
- **Improves competitiveness**: By retaining valuable customers and reducing churn rates, businesses can strengthen their position in the market and sustain long-term growth.
## Data

**Source Link:** https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data

**Preprocessing**:

1. Drop rows with NaN values in any column, replace categorical values with numeric codes, and convert float date columns to datetime format.
2. Optionally drop the 'msno' column.
3. Reorder columns to move the target variable 'is_churn' 
towards the end of the DataFrame.


**Characteristics:**

Object Type: 3 (msno, gender, registration_init_time)

Integer Type: 1 (is_churn)

Float Type: 19 (city, bd, registered_via, payment_method_id, payment_plan_days, plan_list_price, actual_amount_paid, is_auto_renew, transaction_date, membership_expire_date, is_cancel, date, num_25, num_50, num_75, num_985, num_100, num_unq, total_secs)
## Features

- **msno**: User ID, a unique identifier for each user in the dataset.
- **is_churn**: Target variable indicating whether a user churned (did not renew subscription within 30 days of expiration) or renewed (is_churn = 1 for churn, 0 for renewal).
- **transactions**: Information about users' transactions, including payment method, length of membership plan, plan price, actual amount paid, auto-renewal status, transaction date, membership expiration date, and cancellation status.
- **user_logs**: Daily user logs describing listening behaviors, including the number of songs played at different percentages of song length, number of unique songs played, and total seconds played.
- **user_information**: Demographic information about users, including city, age (with outlier values), gender, registration method, registration initiation time, and expiration date (snapshot of member data extraction, not actual churn behavior representation).


## Technologies Used:

- **Python**: The primary programming language used for developing the project and implementing machine learning algorithms.
- **scikit-learn**: A powerful machine learning library in Python used for building predictive models, including RandomForestClassifier for churn prediction.
- **pandas**: A data manipulation and analysis library in Python used for handling the dataset, performing data preprocessing, and exploratory data analysis (EDA).
- **matplotlib**: A data visualization library in Python used for creating visualizations to better understand the data and model performance.


## Model

The function train_evaluate_neural_network trains and evaluates a feedforward neural network for binary classification tasks. Here's a concise overview:

**Model Architecture:**

The model is a feedforward neural network with ReLU activation in the hidden layers and sigmoid activation in the output layer, chosen for its flexibility in capturing complex patterns in data, mitigating overfitting through regularization techniques, and offering better adaptability to non-linear relationships, which is advantageous for binary classification tasks compared to Random Forest or logistic regression.

## Results

Log Loss: 0.0757

Performance Metrics:
- Accuracy: 0.9763
- Precision: 0.9039
- Recall: 0.9001
- F1 Score: 0.9020

Confusion Matrix:
 
     [[167198   2238]    
     [ 2338    21059]]


Classification Report:
        
                precision   recall   f1-score     support

           0       0.99      0.99      0.99        169436

           1       0.90      0.90      0.90        23397

    accuracy                           0.98        192833

    macro avg      0.95      0.94      0.94        192833

    weighted avg   0.98      0.98      0.98        192833
