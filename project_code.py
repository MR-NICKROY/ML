
# import pandas as pd
# import numpy as np
# import joblib

# # ================= LOAD MODEL ASSETS =================
# model = joblib.load("fraud_model.pkl")
# encoder = joblib.load("category_encoder.pkl")
# model_features = joblib.load("model_features.pkl")


# def preprocess_and_predict(input_data: dict):
#     """
#     input_data: dict containing raw transaction data
#     """

#     # Convert input to DataFrame
#     df = pd.DataFrame([input_data])

#     # ================= FEATURE ENGINEERING =================

#     # Time features
#     df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
#     df["LastLogin"] = pd.to_datetime(df["LastLogin"], errors="coerce")

#     df["gap"] = (df["Timestamp"] - df["LastLogin"]).dt.days.abs().fillna(0)
#     df["Hour"] = df["Timestamp"].dt.hour.fillna(0)

#     # Benford's Law
#     def first_digit(x):
#         s = str(abs(x)).replace(".", "").lstrip("0")
#         return int(s[0]) if s else 0

#     df["Benford_Prob"] = df["TransactionAmount"].apply(first_digit).map(
#         {d: np.log10(1 + 1/d) for d in range(1, 10)}
#     ).fillna(0)

#     # Collusion placeholders (backend/db can replace later)
#     df["Transaction_Frequency"] = input_data.get("Transaction_Frequency", 1)
#     df["Total_Linked_Value"] = input_data.get("Total_Linked_Value", df["TransactionAmount"])

#     # Suspicious flag
#     df["SuspiciousFlag"] = input_data.get("SuspiciousFlag", 0)

#     # Category encoding
#     try:
#         df["Category"] = encoder.transform(df["Category"])
#     except ValueError:
#         df["Category"] = -1   # safer than 0

#     # ================= ALIGN FEATURES =================
#     for col in model_features:
#         if col not in df.columns:
#             df[col] = 0

#     df_final = df[model_features]

#     # ================= PREDICTION =================
#     pred = model.predict(df_final)[0]
#     prob = model.predict_proba(df_final)[0][1]

#     return {
#         "is_fraud": bool(pred),
#         "risk_score": float(prob)
#     }


# # ================= TEST =================
# if __name__ == "__main__":
#     sample = {
#         "TransactionAmount": 55000,
#         "Timestamp": "2025-01-01 12:00:00",
#         "LastLogin": "2024-12-20 10:00:00",
#         "Category": "Travel",
#         "AnomalyScore": 0.9,
#         "Transaction_Frequency": 2,
#         "Total_Linked_Value": 120000,
#         "SuspiciousFlag": 1
#     }

#     print(preprocess_and_predict(sample))


# #------------------IMPORT NECESSARY LIBRARIES--------------------

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import warnings as wr
# wr.filterwarnings(action="ignore")
# import seaborn as sns

# #-------------------LOAD THE DATASETS--------------------------------
# DATA_PATH = "data/"

# account    = pd.read_csv(DATA_PATH + "account_activity.csv")
# customer   = pd.read_csv(DATA_PATH + "customer_data.csv")
# fraud      = pd.read_csv(DATA_PATH + "fraud_indicators.csv")
# suspision  = pd.read_csv(DATA_PATH + "suspicious_activity.csv")
# merchant   = pd.read_csv(DATA_PATH + "merchant_data.csv")
# tran_cat   = pd.read_csv(DATA_PATH + "transaction_category_labels.csv")
# amount     = pd.read_csv(DATA_PATH + "amount_data.csv")
# anamoly    = pd.read_csv(DATA_PATH + "anomaly_scores.csv")
# tran_data  = pd.read_csv(DATA_PATH + "transaction_metadata.csv")
# tran_rec   = pd.read_csv(DATA_PATH + "transaction_records.csv")

# data = [account,customer,fraud,suspision,merchant,tran_cat,amount,anamoly,tran_data,tran_rec]
# for df in data:
#     print(df.head())

# #--------------- Merging Customer Data-------------
# costumer_data = pd.merge(customer, account, on='CustomerID')
# costumer_data = pd.merge(costumer_data, suspision, on='CustomerID')
# print(costumer_data)

# #----------------Merging Transaction Data------------
# transaction_data1 = pd.merge(fraud, tran_cat, on="TransactionID")
# transaction_data2 = pd.merge(amount, anamoly, on="TransactionID")
# transaction_data3 = pd.merge(tran_data, tran_rec, on="TransactionID")
# transaction_data = pd.merge(transaction_data1, transaction_data2,on="TransactionID")
# transaction_data = pd.merge(transaction_data, transaction_data3,on="TransactionID")
# print(transaction_data)

# data = pd.merge(transaction_data, costumer_data,on="CustomerID")
# print(data)

# #-------------------EXPLORING THE DATA-------------------------

# print(data.info())
# print(data.shape)
# print(data.describe())

# print(data.columns)

# numerical_features = data.select_dtypes(include=['number']).columns.tolist()
# categorical_features = data.select_dtypes(include=['object']).columns.tolist()
# print(numerical_features)
# print(categorical_features)

# #----------------DATA VISUALIZATION----------------------
# for column in data.columns:
#     if data[column].dtype == 'object':  # Check if the column has a categorical data type
#         top_10_values = data[column].value_counts().head(10)  # Get the first 10 unique values and their counts
#         plt.figure(figsize=(10, 5))  # Adjust the figure size if needed
#         sns.countplot(x=column, data=data, order=top_10_values.index)
#         plt.title(f'Count Plot for {column}')
#         plt.xticks(rotation=90)  # Rotate x-axis labels if they are long
#         plt.show()


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# # Assuming 'data' is your DataFrame containing numerical columns

# # Get the number of numerical columns
# num_cols = len(data.select_dtypes(include=['number']).columns)

# # Calculate the number of rows and columns for subplots
# num_rows = (num_cols // 2) + (num_cols % 2)

# # Create subplots
# fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6*num_rows))
# fig.suptitle("Box Plots for Numerical Columns")

# # Loop through the numerical columns and create box plots
# for i, column in enumerate(data.select_dtypes(include=['number']).columns):
#     row = i // 2
#     col = i % 2
#     sns.boxplot(x=data[column], ax=axes[row, col])
#     axes[row, col].set_title(column)
# # Remove any empty subplots
# if num_cols % 2 != 0:
#     fig.delaxes(axes[num_rows-1, 1])

# plt.tight_layout()
# plt.subplots_adjust(top=0.95)  # Adjust the position of the overall title
# plt.show()

# # We should use countplot for SuspiciousFlag feature

# plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
# sns.countplot(x='SuspiciousFlag', data=data, palette='Set2')  # You can change the palette as desired
# plt.title('Count Plot for Suspicious Flag')
# plt.xlabel('Suspicious Flag')
# plt.ylabel('Count')
# plt.xticks(rotation=45)  # Rotate x-axis labels if they are long

# plt.show()

# #As we can see the dataset's target feature is heavily imbalanced so we can use further techiniqes to equalize the feature's values

# # Select only the numeric columns
# numeric_data = data.select_dtypes(include=['number'])

# # Calculate the correlation matrix for numeric columns
# correlation_matrix = numeric_data.corr()

# # Create a heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap for Numeric Columns')

# plt.show()

# # Dropping the columns as of now they are not mush corelated & also wouldn't damper the performance of model
# from sklearn.model_selection import cross_val_score

# #------------------COLLUSION------------------

# # Create unique Link ID between Customer and Merchant
# data['Link_ID'] = data['CustomerID'].astype(str) + "_" + data['MerchantID'].astype(str)

# # Feature 1: How often do these two transact?
# data['Transaction_Frequency'] = data.groupby('Link_ID')['TransactionAmount'].transform('count')

# # Feature 2: What is the total volume of money moving between them?
# data['Total_Linked_Value'] = data.groupby('Link_ID')['TransactionAmount'].transform('sum')


# #---------------Benford's Law Application-----------------

# # Extract leading digit and map to Benford Probability
# def get_first_digit(amount):
#     s = str(abs(amount)).replace('.', '').lstrip('0')
#     return int(s[0]) if s else 0

# data['First_Digit'] = data['TransactionAmount'].apply(get_first_digit)
# benford_dist = {d: np.log10(1 + 1/d) for d in range(1, 10)}
# data['Benford_Prob'] = data['First_Digit'].map(benford_dist).fillna(0)
# columns_to_be_dropped = ['TransactionID','MerchantID','CustomerID','Name', 'Age', 'Address']

# data1 = data.drop(columns_to_be_dropped, axis=1)
# print(data1.head())

# print(data1['FraudIndicator'].value_counts(), data1['SuspiciousFlag'].value_counts(), data1['Category'].value_counts())

# #--------------------------FEATURE ENGINEERING-------------------------

# # Using Feature Engineering Creating two Columns
# # Hour of Transaction = hour
# # Gap between the day of transaction and last login in days = gap
# if pd.api.types.is_datetime64_any_dtype(data['Timestamp']):
#     print("The 'Timestamp' column is already in datetime format.")
# else:
#     print("The 'Timestamp' column is not in datetime format.")

# data1['Timestamp1'] = pd.to_datetime(data1['Timestamp'])

# print(data1.dtypes)
# data1['Hour'] = data1['Timestamp1'].dt.hour
# data1['LastLogin'] = pd.to_datetime(data1['LastLogin'])
# data1['gap'] = (data1['Timestamp1'] - data1['LastLogin']).dt.days.abs()

# print(data1.head())

# #------------------DATA MODELLING----------------------------------
# X = data1.drop(['FraudIndicator','Timestamp','Timestamp1','LastLogin'],axis=1)
# Y = data1['FraudIndicator']


# from sklearn.preprocessing import LabelEncoder
# # Create an instance of LabelEncoder
# label_encoder = LabelEncoder()

# # Fit and transform the 'Category' column
# X['Category'] = label_encoder.fit_transform(X['Category'])
# print(X)

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

# print(X_train.shape, Y_test.shape)

# # Logistic Regression model

# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# # # Create a logistic regression model
# # model = DecisionTreeClassifier()

# # # Train the model on the training data
# # model.fit(X_train, Y_train)

# # # Make predictions on the testing data
# # y_pred = model.predict(X_test)

# # # Calculate the accuracy of the model
# # accuracy = accuracy_score(Y_test, y_pred)
# # print("Accuracy:", accuracy)

# # #High accuracy is not often a good thing in a machine learning model as it states the problem of imbalanced dataset

# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming you have a DataFrame 'df' with a 'FraudIndicator' column
# # Load your data into the DataFrame if not already done

# # Create a count plot for the 'FraudIndicator' column
# plt.figure(figsize=(8, 6))  # Optional: Adjust the figure size
# sns.countplot(data=data1, x='FraudIndicator', palette='viridis')
# plt.title('Count Plot of Fraud Indicator')
# plt.xlabel('Fraud Indicator')
# plt.ylabel('Count')
# print(plt.show())


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from imblearn.over_sampling import SMOTE
# from collections import Counter


# # Initialize SMOTE for oversampling
# smote = SMOTE(random_state=42)

# # Apply SMOTE to the data
# X_resampled, y_resampled = smote.fit_resample(X, Y)

# # Check the class distribution after oversampling
# print("Class distribution after oversampling:", Counter(y_resampled))

# # Create a count plot for the 'FraudIndicator' column after oversampling
# plt.figure(figsize=(8, 6))
# sns.countplot(data=pd.DataFrame({'FraudIndicator': y_resampled}), x='FraudIndicator', palette='viridis')
# plt.title('Count Plot of Fraud Indicator (After Oversampling)')
# plt.xlabel('Fraud Indicator')
# plt.ylabel('Count')
# plt.show()

# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# from sklearn.utils import shuffle

# # Shuffle the dataset to introduce slight randomness (you can adjust the `random_state` for different outcomes)
# X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=42)

# # Split data into train and test sets (keep the test set separate for evaluation)
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report

# # 1. Initialize the Random Forest Classifier
# # We use 100 trees (n_estimators) and a fixed random_state for reproducibility
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# # 2. Train the model on the resampled (balanced) training data
# rf_model.fit(X_train, y_train)

# # 3. Make predictions on the test data
# y_pred_rf = rf_model.predict(X_test)

# # 4. Evaluate the model performance
# # accuracy_rf = accuracy_score(y_test, y_pred_rf)
# # precision_rf = precision_score(y_test, y_pred_rf)
# # recall_rf = recall_score(y_test, y_pred_rf)
# # f1_rf = f1_score(y_test, y_pred_rf)
# # conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# # print("Random Forest Model Evaluation Metrics:")
# # print("Accuracy:",accuracy_rf)
# # print("Precision:", precision_rf)
# # print("Recall:",recall_rf)
# # print("F1 Score:", f1_rf)
# # print("Confusion Matrix:")
# # print(conf_matrix_rf)

# # # Detailed Classification Report
# # print("Detailed Classification Report:")
# # print(classification_report(y_test, y_pred_rf))

# #----------- Cross-Validation to assess model stability----------------

# from sklearn.model_selection import cross_val_score, StratifiedKFold
# import numpy as np

# # 1. Define the Cross-Validation strategy
# # We use StratifiedKFold to ensure each fold has a balanced percentage of fraud
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # 2. Calculate Cross-Validation scores for 'Recall' 
# recall_scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=skf, scoring='recall')

# # 3. Calculate Cross-Validation scores for 'Accuracy'
# accuracy_scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=skf, scoring='accuracy')

# print("--- Cross-Validation Results (5-Folds) ---")
# print(f"Average Recall: {np.mean(recall_scores):.4f} (+/- {np.std(recall_scores):.4f})")
# print(f"Average Accuracy: {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")

# import joblib


# # 'rf_model' is the variable name from your Random Forest training code
# joblib.dump(rf_model, 'fraud_model.pkl')

# # The backend needs this to translate new data exactly the same way.
# joblib.dump(label_encoder, 'category_encoder.pkl')

# # This saves the exact list of columns your model expects (e.g., 'Hour', 'gap', 'Amount')
# model_columns = X.columns.tolist()
# joblib.dump(model_columns, 'model_features.pkl')

# print("Success! Files 'fraud_model.pkl', 'category_encoder.pkl', and 'model_features.pkl' are ready for the backend.")
# # Inference on new/unseen data (for example, use a separate unseen dataset or a specific test sample)
# # Here we simulate it by using the first row from the X_test
# unseen_sample = X_test.iloc[69].values.reshape(1, -1)  # Reshaping for a single sample

# # Predict the label for the unseen sample
# inference_prediction = rf_model.predict(unseen_sample)

# # Map prediction result to 'fraud' or 'not fraud'
# fraud_status = "Fraud" if inference_prediction[0] == 1 else "Not Fraud"

# print("Inference Prediction for Unseen Sample:", fraud_status)