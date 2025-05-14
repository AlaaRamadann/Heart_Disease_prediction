#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Replace '4' with the number of cores you want to use
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")


# In[59]:


df=pd.read_csv("heart_disease_risk_dataset_corrupted.csv")
df


# In[47]:


df.duplicated().sum()


# In[60]:


df.drop_duplicates(inplace=True)


# In[6]:


df.duplicated().sum()


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[61]:


df['Age'] = pd.to_numeric(df['Age'], errors='coerce')


# In[53]:


plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=30, kde=True, color='royalblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.show()


# In[10]:


df['Age'].isnull().sum()


# In[62]:


df['Age']=df['Age'].fillna(df['Age'].median())


# In[63]:


df['Age'].isnull().sum()


# In[64]:


df['High_BP'].value_counts()


# In[65]:


df = df[df['High_BP'] != 2.0]


# In[66]:


df['High_BP'].value_counts()


# In[67]:


imputer = KNNImputer(n_neighbors=15)
df.loc[:, 'High_Cholesterol'] = imputer.fit_transform(df[['High_Cholesterol']]).round().astype(int)
print(df['High_Cholesterol'].value_counts())


# In[68]:


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=15)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
print(df_imputed.isnull().sum())


# In[69]:


df.info()


# In[70]:


df.describe()


# In[19]:


sns.boxplot(df)
plt.xticks(rotation=95)


# In[21]:


sns.boxplot(x=df['Age'])


# In[22]:


(df['Age'] >= 18) & (df['Age'] <= 120)


# In[23]:


df = df[(df['Age'] >= 18) & (df['Age'] <= 120)]
print(df['Age'].describe())


# In[24]:


print(df['Smoking'].value_counts())


# In[71]:


most_frequent_value = df['Smoking'].mode()[0]
df['Smoking'] = df['Smoking'].apply(lambda x: most_frequent_value if x not in [0, 1] else x)


# In[26]:


print(df['Smoking'].value_counts())


# In[27]:


df['Smoking'].describe()


# In[28]:


df.describe()


# In[29]:


df['Heart_Risk'].value_counts().plot(kind='barh', color="#204D00", figsize=(6, 4))
plt.xlabel("Count", labelpad=14)
plt.ylabel("Target Variable", labelpad=14)
plt.title("Count of TARGET Variable per category", y=1.02)


# In[30]:


df['Heart_Risk'].value_counts()


# In[31]:


# Checking which feature/column should be converted into Bins
for column in df.columns:
    print(f'{column} =>', df[column].value_counts().shape)


# In[32]:


bins = [18, 25, 35, 45, 55, float('inf')]
labels = ['18-25', '26-35', '36-45', '46-55', '56+']
df['Age_Binned'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)
print(df['Age_Binned'].value_counts())


# In[33]:


plt.figure(figsize=(8, 5))
sns.countplot(x='Age_Binned', data=df, palette='viridis')
plt.title('Age Group Distribution')
plt.xlabel('Age Groups')
plt.ylabel('Count')
plt.show()


# In[34]:


gender_colors = ['#3498db', '#ff69b4']
risk_colors = ['#87CEFA', '#FFB6C1']

gender_counts = df['Gender'].value_counts()

male_risk = df[df['Gender'] == 1]['Heart_Risk'].value_counts()
female_risk = df[df['Gender'] == 0]['Heart_Risk'].value_counts()

fig, axes = plt.subplots(1, 3, figsize=(10, 4))

axes[0].pie(gender_counts, labels=['Male', 'Female'], autopct='%1.1f%%', colors=gender_colors)
axes[0].set_title('Gender Distribution')

axes[1].pie(male_risk, labels=['No Risk', 'Risk'], autopct='%1.1f%%', colors=risk_colors)
axes[1].set_title('Heart Risk (Male)')

axes[2].pie(female_risk, labels=['No Risk', 'Risk'], autopct='%1.1f%%', colors=risk_colors)
axes[2].set_title('Heart Risk (Female)')

plt.tight_layout()
plt.show()


# In[35]:


age_risk_counts = df.groupby(['Age_Binned', 'Heart_Risk']).size().unstack(fill_value=0)
age_risk_percent = age_risk_counts.div(age_risk_counts.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(7, 4))

colors = ['#66b3ff', '#ff66b3']
labels = ['No Risk', 'Risk']

bars = age_risk_percent.plot(kind='bar', stacked=True, color=colors, ax=ax, edgecolor='black')

for i, (no_risk, risk) in enumerate(zip(age_risk_percent[0], age_risk_percent[1])):
    ax.text(i, no_risk / 2, f'{no_risk:.1f}%', ha='center', color='black', fontsize=10)
    ax.text(i, no_risk + risk / 2, f'{risk:.1f}%', ha='center', color='black', fontsize=10)

ax.set_ylabel('Percentage')
ax.set_xlabel('Age Group')
ax.set_title('Heart Risk Distribution by Age Group')
ax.legend(labels)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[36]:


categorical_columns = ['Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations', 'Dizziness',
                       'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea', 'Diabetes', 'Smoking',
                       'Obesity', 'Sedentary_Lifestyle', 'Family_History', 'Chronic_Stress']

plt.figure(figsize=(25, 30))

for i, col in enumerate(categorical_columns, 1):

    plt.subplot(6,4, i)
    sns.countplot(x='Gender', hue='Heart_Risk', data=df, palette='Set1')
    plt.title(f'Relationship Between {col} and Heart Risk')
    plt.xlabel(col)
    plt.ylabel('Count')

plt.legend(['No Risk (0)', 'Risk (1)'])
plt.show()


# In[37]:


print(df.columns)


# In[20]:


df = df.drop('Age_Binned', axis=1, errors='ignore')


# In[40]:


corr_matrix = df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


# In[72]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['Age'] = scaler.fit_transform(df[['Age']])


# In[19]:


print(df['High_BP'].isnull().sum())


# In[20]:


print(df['High_BP'].unique())
print(df['High_BP'].value_counts())


# In[21]:


print(df['High_BP'].isna().sum())
print(df['High_BP'].isnull().sum())


# In[73]:


# Filling missing values using the mean
df['High_BP'].fillna(df['High_BP'].mean(), inplace=True)

# Filling missing values using the median
# df['High_BP'].fillna(df['High_BP'].median(), inplace=True)

# Checking for missing values after the treatment
print(df['High_BP'].isna().sum())


# In[74]:


from sklearn.model_selection import train_test_split

# Splitting the data into X (features) and y (target)
X = df.drop('Heart_Risk', axis=1)
y = df['Heart_Risk']

# Splitting the data into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Defining the models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
}

# Training the models and evaluating them
for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Make predictions on the test set

    print(f"Model: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")  # Print accuracy
    print(f"Classification Report:\n {classification_report(y_test, y_pred)}")  # Print classification report
    print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")  # Print confusion matrix
    print("="*50)  # Print a separator line


# In[ ]:





# In[76]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define the Logistic Regression model
log_reg = LogisticRegression()

# Specify the parameters to test
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter (C)
    'solver': ['liblinear', 'saga'],  # Solver selection
    'max_iter': [100, 200, 300]  # Maximum number of iterations
}

# Apply GridSearchCV
grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Print the best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate the model using the best parameters
best_log_reg = grid_search.best_estimator_
y_pred = best_log_reg.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy after hyperparameter tuning: {accuracy}")


# In[78]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(best_log_reg, X, y, cv=5)
print(f"Cross-Validation Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")


# In[ ]:


import joblib

# Save the trained model
joblib.dump(best_log_reg, 'final_logistic_regression_model.pkl')


# In[ ]:


# !streamlit run app.py


# In[ ]:


# import mlflow
# import mlflow.sklearn
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import os
# from mlflow import MlflowClient

# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# mlflow.set_experiment("experiment2")


# client = MlflowClient()

# with mlflow.start_run():
#     for name, model in models.items():
#         with mlflow.start_run(nested=True):
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             version=1
#             mlflow.log_param("model_name", name)
#             mlflow.log_param("algorithm", name)

#             accuracy = accuracy_score(y_test, y_pred)
#             mlflow.log_metric("accuracy", accuracy)

#             mlflow.sklearn.log_model(model, name)
#             mlflow.sklearn.log_model(model, "model")

#             # Build model URI from the run ID
#             model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

#             # Register the model to the MLflow Model Registry
#             result = mlflow.register_model(model_uri=model_uri, name=name)
#             version = result.version  # Automatically assigned version 



#             print(f"Model: {name}")
#             print(f"Accuracy: {accuracy}")
#             print(f"Classification Report:\n {classification_report(y_test, y_pred)}")
#             print("="*50)

# mlflow.end_run()


# In[ ]:




