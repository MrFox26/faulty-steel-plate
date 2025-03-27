#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
import pandas as pd
import numpy as np

#for visualization
import matplotlib.pyplot as plt
import seaborn as sn
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import optuna
import xgboost as xgb
import plotly.graph_objects as go


# In[2]:


#pip install xgboost


# In[3]:


#pip install optuna


# In[4]:


# load the dataset
df = pd.read_csv('faults.csv')


# # Basic Understanding of data

# In[6]:


#to know how big dataset is
df.shape


# In[7]:


# to know how data looks like
df.sample(10)


# In[8]:


df.columns


# In[9]:


# here the last 7 columns represent fault classes so i will convert the 7 columns into a single one and remove the unnecessary columns.
# Check if any row has multiple fault labels
target_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

def check_multilabel(df, target_columns):
    multilabel_rows = df[target_columns].sum(axis=1) > 1
    if multilabel_rows.any():
        print("This is a multi-label classification problem.")
        print(f"Total multi-label rows: {multilabel_rows.sum()}")
    else:
        print("This is a multi-class classification problem.")
        
check_multilabel(df,target_columns)


# In[10]:


#converting into single column
df['Fault_type'] = df[target_columns].idxmax(axis = 1)
df=df.drop(target_columns, axis=1)

df.sample(10)


# In[11]:


# to know datatype of columns
df.info()


# In[12]:


# to check missing values
df.isnull().sum()
# here there is no missing values


# In[13]:


# to know how data looks mathematically
df.describe()


# In[14]:


# to know duplicate values
df.duplicated().sum()


# # EDA

# In[16]:


df['Fault_type'].value_counts()


# In[17]:


df.nunique()


# In[18]:


df['TypeOfSteel_A300'].unique()


# In[19]:


df['TypeOfSteel_A400'].unique()


# In[20]:


df['Outside_Global_Index'].unique()


# In[21]:


# Rename multiple columns
df.rename(columns={'TypeOfSteel_A300': 'A300', 'TypeOfSteel_A400': 'A400'}, inplace=True)

#converting the 2 type of steel column into a single one
type_of_steel = ['A300','A400']
df['Steel_type'] = df[type_of_steel].idxmax(axis = 1)
df=df.drop(type_of_steel, axis=1)


# In[22]:


df.sample(5)


# In[23]:


df['Steel_type'].value_counts()


# In[24]:


# Outliers
from sklearn.ensemble import IsolationForest

# Drop categorical columns and ensure data is numerical
categorical_col = ['Fault_type', 'Steel_type']
dfn = df.drop(columns=categorical_col)

# Instantiate Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)  # Set contamination rate (e.g., 5% of data assumed to be outliers)

# Fit the model and predict
outlier_predictions = iso_forest.fit_predict(dfn)

# IsolationForest returns:
#  1 for inliers
# -1 for outliers
df['Outlier'] = outlier_predictions

# Count and display outliers
num_outliers = (df['Outlier'] == -1).sum()
print(f"Number of outliers detected: {num_outliers}")

# Filter outliers and inliers
outliers = df[df['Outlier'] == -1]
inliers = df[df['Outlier'] == 1]

# visualizing outlier
fig = px.box(
    dfn,
    orientation="h",  # Horizontal orientation to match the original visualization
    title="Interactive Boxplot to Visualize Outliers in Each Feature",
    template="plotly",  # Default Plotly theme
    color_discrete_sequence=["#2ca02c"],  # Set a custom color
)
fig.update_layout(
    title_font=dict(size=20),
    xaxis_title="Values",
    yaxis_title="Features",
    xaxis=dict(showgrid=True, title_font=dict(size=16)),
    yaxis=dict(showgrid=False, title_font=dict(size=16)),
)
# Display the plot
fig.show()


# removing outliers
dfc = df[df['Outlier'] == 1]


# In[25]:


dfc['Outlier'].unique()


# In[26]:


dfc = dfc.drop(columns=['Outlier'])


# # visualization
# 

# In[28]:


import plotly.express as px

# Create an interactive count plot for multiple categorical columns
fig = px.bar(
    dfc,
    x="Fault_type",  # First categorical column
    color="Steel_type",  # Second categorical column for coloring
    title="Count of Fault Types by Steel Type",
    text_auto=True,  # Display counts on the bars
    template="plotly",  # Default theme
    barmode="group"  # Group bars for each Fault_type by Steel_type
)

# Update layout for better presentation
fig.update_layout(
    title_font=dict(size=20),
    xaxis_title="Fault Type",
    yaxis_title="Count",
    xaxis=dict(title_font=dict(size=16)),
    yaxis=dict(title_font=dict(size=16)),
)

# Display the plot
fig.show()


# In[29]:


categorical_col = ['Fault_type','Steel_type']
dfcn = dfc.drop(columns =categorical_col)

for column in df.columns:
    # Create a histogram for each feature with KDE (Kernel Density Estimate) overlay
    fig = px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}",
        nbins=30,  
        marginal="box", 
        color_discrete_sequence=["skyblue"],  
        histnorm="probability density" 
    )
    
    # Show the plot
    fig.show()


# In[30]:


#correlation matrix
corr_matrix = dfcn.corr()
plt.figure(figsize=(20, 20))
sn.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

threshold = 0.9
# Create a mask to ignore the upper triangle and self-correlations
mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

# Extract the pairs that meet the correlation threshold
strong_corr_pairs = corr_matrix.where(mask).stack().reset_index()

# Rename columns for clarity
strong_corr_pairs.columns = ['Feature1', 'Feature2', 'Correlation']

# Filter pairs based on the threshold
strong_corr_pairs = strong_corr_pairs[(strong_corr_pairs['Correlation'] > threshold) | (strong_corr_pairs['Correlation'] < -threshold)]

# Display strong correlation pairs
print("Strongly correlated feature pairs:")
print(strong_corr_pairs)


# In[31]:


dfc.info()


# In[32]:


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
le = LabelEncoder()
dfc['Fault_type'] = le.fit_transform(dfc['Fault_type'])

#Check the mapping
print("Fault_type classes:", dict(zip(le.classes_, le.transform(le.classes_))))

dfc['Steel_type'] = le.fit_transform(dfc['Steel_type'])



# # Modelling
# 

# In[34]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def model_pipeline(model):
    return Pipeline(steps=[
        ('scaler', MinMaxScaler()),  
        ('classifier', model)       
    ])

# Splitting dataset
X = dfc.drop(['Fault_type'], axis=1)
y = dfc['Fault_type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models
models = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'KNeighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss') 
}

# Train and evaluate models
for name, model in models.items():
    pipeline = model_pipeline(model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')  

    print(f"\n{name} Model Performance:")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(classification_report(y_test, y_pred))


# In[35]:


import joblib


# In[37]:


joblib.dump(model, "xgboost.pkl")

print("Model saved successfully!")


# In[ ]:




