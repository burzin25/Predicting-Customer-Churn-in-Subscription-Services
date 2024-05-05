## If there are some libraries missing run the below code in VS Code Terminal
#pip install -r requirements.txt

# Import Libraries
import src.read as read
import src.preprocess as pre
import src.eda as eda
import src.feature_engg_sel as fs
import src.train_model as tm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
import os


# Get the directory where main.py is located
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct file paths relative to the current directory
file_paths = {
    "members": os.path.join(current_directory, "data/members_v3.csv"),
    "train": os.path.join(current_directory, "data/train_v2.csv"),
    "transactions": os.path.join(current_directory, "data/transactions_v2.csv"),
    "logs": os.path.join(current_directory, "data/user_logs_v2.csv")
}


joined_data_frame = read.read_and_join_csv_files(file_paths)

# Preprocess the dataframe
preprocessed_df = pre.preprocessing(joined_data_frame)

# Explore Data
columns_to_plot = ['city', 'gender', 'registered_via', 'payment_method_id', 'is_auto_renew', 'is_cancel', 'is_churn']
eda.count_plot(preprocessed_df, columns_to_plot)

eda.plot_is_churn(preprocessed_df)

# Undersampling the data
undersampled_df = pre.adjust_data_quantity(preprocessed_df, churn_percent = 50, data_size_percent = 100)

# Feature Engineering
featurized_df = fs.feature_engineering(undersampled_df)

# Preprocess the dataframe again
preprocessed_v2_df = pre.preprocessing_v2(featurized_df)
outliers_df = pre.drop_outliers(preprocessed_v2_df)
large_val_df = pre.drop_rows_with_large_values(outliers_df)

# Separating Dataset into independent and depandent feature dataframes
X = large_val_df.drop(columns=['is_churn'])  # Independent variables (features)
y = large_val_df['is_churn']  # Dependent variable (target)

# Feature Selection
selected_features = fs.select_best_features(X,y,k=10,score_func=f_classif)
X_selected_df = X[selected_features[:10]]
eda.correlation_plot(X_selected_df, y)

# Training and Modeling
X_train, X_val, X_test, y_train, y_val, y_test = tm.split_dataset(X, y, test_size=0.2, val_size=0.2, random_state=42)

scaler = MinMaxScaler()  # or StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

tm.train_evaluate_neural_network(X_train_normalized, y_train, X_test_normalized, y_test, epochs=25, batch_size=64, validation_split=0.2, verbose=1)