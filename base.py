import pandas as pd
import numpy as np
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
from datetime import datetime

# Streamlit App Configuration
st.set_page_config(page_title="Banking Security: Deadlock Detection", layout="wide")

# User Authentication (Simulated)
st.sidebar.title("Login")
username = st.sidebar.text_input("Username", key="username")
password = st.sidebar.text_input("Password", type="password", key="password")

if username == "admin" and password == "admin123":
    st.sidebar.success("Welcome, Admin!")
else:
    st.sidebar.error("Invalid Credentials! Please log in.")
    st.stop()

# System Configuration
st.sidebar.title("Configuration")
batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=50, value=10)
model_type = st.sidebar.selectbox("Model", options=["Random Forest", "Decision Tree"])
threshold = st.sidebar.slider("Deadlock Detection Threshold", min_value=0.0, max_value=1.0, value=0.5)

# Simulated Transaction Data
np.random.seed(42)
n_transactions = 300

data = {
    'Transaction_ID': [f"T_{i}" for i in range(1, n_transactions + 1)],
    'Process_ID': np.random.choice([f"P_{i}" for i in range(1, 10)], n_transactions),
    'Resource_Aquired': np.random.choice(['R_A', 'R_B', 'R_C', 'R_D'], n_transactions),
    'Resource_Waiting': np.random.choice(['R_A', 'R_B', 'R_C', 'R_D'], n_transactions),
    'Status': np.random.choice(['Running', 'Waiting'], n_transactions),
    'Timestamp': pd.date_range("2024-11-01", periods=n_transactions, freq='H')
}

df_transactions = pd.DataFrame(data)
df_transactions['Deadlock_Label'] = 0

# Ensure the 'Timestamp' column is in datetime format
df_transactions['Timestamp'] = pd.to_datetime(df_transactions['Timestamp'])

# Deadlock Detection Function
def detect_deadlocks(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        txn = row['Transaction_ID']
        res_wait = row['Resource_Waiting']
        res_acq = row['Resource_Aquired']

        if row['Status'] == 'Waiting':
            G.add_edge(txn, res_wait)
        if row['Status'] == 'Running':
            G.add_edge(res_acq, txn)

    cycles = list(nx.simple_cycles(G))
    deadlocks = [cycle for cycle in cycles if len(cycle) > 1]
    return deadlocks, G

# Label Deadlocks in Data
deadlocks, G = detect_deadlocks(df_transactions)
for cycle in deadlocks:
    for transaction in cycle:
        if transaction.startswith('T_'):
            df_transactions.loc[df_transactions['Transaction_ID'] == transaction, 'Deadlock_Label'] = 1

# Machine Learning Pipeline
X = pd.get_dummies(df_transactions[['Process_ID', 'Resource_Aquired', 'Resource_Waiting']])
y = df_transactions['Deadlock_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(random_state=42) if model_type == "Random Forest" else DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Model Evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display Results
st.title("Banking Security: Deadlock Detection")

st.subheader("Transaction Data Overview")
st.dataframe(df_transactions)

st.subheader("Detected Deadlocks")
st.write(f"Number of Deadlocks Detected: {len(deadlocks)}")
st.write(f"Detected Cycles: {deadlocks}")

st.subheader("Model Evaluation")
st.write(f"Model Type: {model_type}")
st.write(f"Accuracy: {accuracy:.2f}")
st.write("Classification Report:")
st.json(report)

st.write("Confusion Matrix:")
st.write(conf_matrix)

# Plot Confusion Matrix
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Deadlock', 'Deadlock'], yticklabels=['No Deadlock', 'Deadlock'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
st.pyplot(fig)

# Feature Importance
if model_type == "Random Forest":
    importance = clf.feature_importances_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)

    st.subheader("Feature Importance")
    st.bar_chart(feature_importance.set_index('Feature'))

# Interactive Visualizations
st.subheader("Transaction Timeline")
fig_timeline = px.timeline(
    df_transactions,
    x_start="Timestamp",
    x_end="Timestamp",
    y="Transaction_ID",
    color="Deadlock_Label",
    title="Transaction Timeline with Deadlocks"
)

# Check if the figure can be serialized to JSON before plotting
try:
    st.plotly_chart(fig_timeline)
except TypeError as e:
    st.error(f"Error: {e}")

st.subheader("Resource Allocation Network")
fig_network = plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=5000, node_color='lightblue', font_size=10, font_color='black', edge_color='gray')
st.pyplot(fig_network)
