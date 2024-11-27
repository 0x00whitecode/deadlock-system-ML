import pandas as pd
import numpy as np
import networkx as nx
import time
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly.graph_objects as go

# Set up Streamlit layout
st.set_page_config(page_title="Advanced Deadlock Detection System", layout="wide")

# Sidebar - Aim and Objectives
st.sidebar.title("System Overview")
st.sidebar.write("""
### Aim
Develop and evaluate a robust deadlock detection system for the banking sector.

### Objectives
1. **Analyze** limitations of current deadlock detection methods.
2. **Design** an advanced deadlock detection framework.
3. **Compare** the new system’s performance with existing methods.
4. **Recommend** integration strategies and future improvements.
""")

# Sidebar - User Inputs for Configuration
st.sidebar.subheader("System Configuration")
batch_size = st.sidebar.slider("Set Batch Size for Real-Time Monitoring", min_value=1, max_value=20, value=5)
model_type = st.sidebar.selectbox("Choose Machine Learning Model", options=["Random Forest", "Decision Tree"])

# Simulate transaction data for a banking system
np.random.seed(42)
n_transactions = 200

data = {
    'Transaction_ID': [f"T_{i}" for i in range(1, n_transactions + 1)],
    'Process_ID': np.random.choice([f"P_{i}" for i in range(1, 10)], n_transactions),
    'Resource_Aquired': np.random.choice(['R_A', 'R_B', 'R_C', 'R_D'], n_transactions),
    'Resource_Waiting': np.random.choice(['R_A', 'R_B', 'R_C', 'R_D'], n_transactions),
    'Status': np.random.choice(['Running', 'Waiting'], n_transactions),
    'Timestamp': pd.date_range("2024-11-01", periods=n_transactions, freq='T')
}

df_transactions = pd.DataFrame(data)
df_transactions['Deadlock_Label'] = 0

# Initialize graph for resource dependencies
G = nx.DiGraph()


# Detect deadlocks function
def detect_deadlocks(df):
    G.clear()
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
    return deadlocks


# Label deadlocks in data
deadlocks = detect_deadlocks(df_transactions)
for cycle in deadlocks:
    for transaction in cycle:
        if transaction.startswith('T_'):
            df_transactions.loc[df_transactions['Transaction_ID'] == transaction, 'Deadlock_Label'] = 1

# Prepare data for machine learning
X = pd.get_dummies(df_transactions[['Process_ID', 'Resource_Aquired', 'Resource_Waiting']])
y = df_transactions['Deadlock_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the chosen model
if model_type == "Random Forest":
    clf = RandomForestClassifier(random_state=42)
else:
    clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

# Model evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

# Dashboard Section
st.title("Banking Deadlock Detection System Dashboard")

# Top-level Metrics
st.subheader("System Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", f"{accuracy:.2%}")
col2.metric("Deadlocks Detected", len(deadlocks))
col3.metric("Batch Size", batch_size)

# Display Classification Report
st.subheader("Classification Report")
st.write(pd.DataFrame(report).transpose())

# Confusion Matrix Visualization
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Deadlock", "Deadlock"],
            yticklabels=["No Deadlock", "Deadlock"], ax=ax)
st.pyplot(fig)

# Interactive Feature Importance
st.subheader("Feature Importance")
perm_importance = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': perm_importance.importances_mean})
fig = px.bar(feature_importance.sort_values(by="importance", ascending=False), x='importance', y='feature',
             orientation='h', title="Feature Importance")
st.plotly_chart(fig)

# Transaction Dependency Graph Visualization
st.subheader("Transaction Dependency Graph")
deadlock_cycles = detect_deadlocks(df_transactions)
if deadlock_cycles:
    for cycle in deadlock_cycles:
        edge_trace = go.Scatter(
            x=[df_transactions.loc[df_transactions['Transaction_ID'] == node].index[0] for node in cycle],
            y=[0] * len(cycle),
            mode="lines+markers+text",
            text=cycle,
            marker=dict(size=10, color="Red"),
            line=dict(width=2, color="Blue"),
        )
        fig_graph = go.Figure(data=[edge_trace])
        st.plotly_chart(fig_graph)
else:
    st.write("No deadlocks detected in current batch")

# Deadlock Timeline Heatmap
st.subheader("Deadlock Frequency Over Time")
deadlock_trends = df_transactions.groupby(df_transactions['Timestamp'].dt.date)['Deadlock_Label'].sum().reset_index()
fig = px.density_heatmap(deadlock_trends, x="Timestamp", y="Deadlock_Label", title="Deadlock Frequency Heatmap")
st.plotly_chart(fig)

# Real-Time Monitoring Simulation
st.subheader("Real-Time Monitoring Simulation")
for i in range(0, len(df_transactions), batch_size):
    batch_data = df_transactions.iloc[i:i + batch_size]
    batch_deadlocks = detect_deadlocks(batch_data)

    if batch_deadlocks:
        st.write(f"Batch {i // batch_size + 1}: Deadlock Detected in {batch_deadlocks}")
    else:
        st.write(f"Batch {i // batch_size + 1}: No Deadlock Detected")

    # Simulate real-time delay
    time.sleep(1)

# Footer
st.sidebar.write("Developed by [Your Name]")
