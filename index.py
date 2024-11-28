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
import plotly.graph_objects as go
from datetime import datetime

# Streamlit App Configuration
st.set_page_config(page_title="Banking Security: Deadlock Detection", layout="wide")

# Sidebar Configuration for User Authentication
def authenticate_user():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username", key="username")
    password = st.sidebar.text_input("Password", type="password", key="password")
    
    if username == "admin" and password == "admin123":
        st.sidebar.success("Welcome, Admin!")
        return True
    else:
        st.sidebar.error("Invalid Credentials! Please log in.")
        return False

# Check user authentication
if not authenticate_user():
    st.stop()

# Sidebar Configuration for System Parameters
def configure_system():
    st.sidebar.title("System Configuration")
    batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=50, value=10)
    model_type = st.sidebar.selectbox("Select Model", options=["Random Forest", "Decision Tree"])
    threshold = st.sidebar.slider("Deadlock Detection Threshold", min_value=0.0, max_value=1.0, value=0.5)
    return batch_size, model_type, threshold

batch_size, model_type, threshold = configure_system()

# Function to Generate Simulated Transaction Data
def generate_transaction_data(n_transactions=300, n_processes=10, deadlock_rate=0.05):
    np.random.seed(42)
    transactions = [f"T_{i}" for i in range(1, n_transactions + 1)]
    processes = [f"P_{i}" for i in range(1, n_processes + 1)]
    resources = ['R_A', 'R_B', 'R_C', 'R_D']
    transaction_types = ['Read', 'Write', 'Update']
    priority_levels = ['Low', 'Medium', 'High']
    
    data = {
        'Transaction_ID': np.random.choice(transactions, n_transactions),
        'Process_ID': np.random.choice(processes, n_transactions),
        'Resource_Aquired': np.random.choice(resources, n_transactions),
        'Resource_Waiting': np.random.choice(resources, n_transactions),
        'Status': np.random.choice(['Running', 'Waiting'], n_transactions),
        'Timestamp': pd.date_range("2024-11-01", periods=n_transactions, freq='H') + pd.to_timedelta(np.random.randint(0, 3600, n_transactions), unit='s'),
        'Transaction_Duration': np.random.randint(1, 300, n_transactions),
        'Resource_Request_Type': np.random.choice(['Read', 'Write'], n_transactions),
        'Priority_Level': np.random.choice(priority_levels, n_transactions),
        'Transaction_Type': np.random.choice(transaction_types, n_transactions),
        'Resource_Queue_Position': np.random.randint(1, 10, n_transactions)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce deadlocks
    num_deadlocks = int(n_transactions * deadlock_rate)
    deadlock_transactions = np.random.choice(df['Transaction_ID'].unique(), num_deadlocks, replace=False)
    deadlock_data = df[df['Transaction_ID'].isin(deadlock_transactions)]
    
    # Create circular deadlocks
    for i in range(0, len(deadlock_data) - 1, 2):
        df.loc[df['Transaction_ID'] == deadlock_data.iloc[i]['Transaction_ID'], 'Resource_Waiting'] = deadlock_data.iloc[i + 1]['Resource_Aquired']
        df.loc[df['Transaction_ID'] == deadlock_data.iloc[i + 1]['Transaction_ID'], 'Resource_Waiting'] = deadlock_data.iloc[i]['Resource_Aquired']
        df.loc[df['Transaction_ID'] == deadlock_data.iloc[i]['Transaction_ID'], 'Status'] = 'Waiting'
        df.loc[df['Transaction_ID'] == deadlock_data.iloc[i + 1]['Transaction_ID'], 'Status'] = 'Waiting'
    
    # Add Deadlock Label
    df['Deadlock_Label'] = df['Transaction_ID'].apply(lambda x: 1 if x in deadlock_transactions else 0)
    
    return df

# Generate Transaction Data
df_transactions = generate_transaction_data(n_transactions=300, n_processes=10, deadlock_rate=0.05)

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
    deadlocks = [cycle for cycle in cycles if len(cycle) > 1]  # Filter out deadlocks
    return deadlocks, G

deadlocks, G = detect_deadlocks(df_transactions)

# Update Deadlock Labels in the DataFrame
for cycle in deadlocks:
    for txn in cycle:
        df_transactions.loc[df_transactions['Transaction_ID'] == txn, 'Deadlock_Label'] = 1

# Prepare Data for Machine Learning
X = pd.get_dummies(df_transactions[['Process_ID', 'Resource_Aquired', 'Resource_Waiting']])
y = df_transactions['Deadlock_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Select Classifier Based on User Choice
clf = RandomForestClassifier(random_state=42) if model_type == "Random Forest" else DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Model Evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display Results on Streamlit
st.title("Banking Security: Deadlock Detection")

# Transaction Data Overview
st.subheader("Transaction Data Overview")
st.dataframe(df_transactions)

# Deadlock Detection Summary
st.subheader("Detected Deadlocks")
st.write(f"Number of Deadlocks Detected: {len(deadlocks)}")
st.write(f"Detected Cycles: {deadlocks}")

# Visualize the Resource Allocation Network (Deadlock View) in 3D
def plot_deadlock_network_3d(G):
    node_trace = go.Scatter3d(
        x=[], y=[], z=[], text=[], mode='markers+text',
        marker=dict(size=10, color=[], opacity=0.8), hoverinfo='text'
    )
    edge_trace = go.Scatter3d(
        x=[], y=[], z=[], line=dict(width=1, color='white'),
        hoverinfo='none', mode='lines'
    )
    
    # Generate 3D layout positions using spring layout
    pos = nx.spring_layout(G, dim=3)  # 3D layout
    for node, (x, y, z) in pos.items():
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['z'] += (z,)
        node_trace['marker']['color'] += ('blue' if 'R_' in node else 'orange',)
        node_trace['text'] += (node,)
    
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
        edge_trace['z'] += (z0, z1, None)
    
    fig_network = go.Figure(data=[edge_trace, node_trace])
    fig_network.update_layout(
        title="3D Resource Allocation Network (Deadlock View)",
        showlegend=False, hovermode='closest',
        scene=dict(xaxis=dict(showgrid=False, zeroline=False),
                   yaxis=dict(showgrid=False, zeroline=False),
                   zaxis=dict(showgrid=False, zeroline=False)),
        plot_bgcolor='white'
    )
    st.plotly_chart(fig_network)

plot_deadlock_network_3d(G)

# Model Evaluation
st.subheader("Model Evaluation")
st.write(f"Model Type: {model_type}")
st.write(f"Accuracy: {accuracy:.2f}")
st.write("Classification Report:")
st.json(report)

# Confusion Matrix
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Deadlock', 'Deadlock'], yticklabels=['No Deadlock', 'Deadlock'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# Transaction Counts Visualization
st.subheader("Transaction Counts by Status")
status_counts = df_transactions['Status'].value_counts()
fig_status = go.Figure(data=[go.Bar(x=status_counts.index, y=status_counts.values)])
fig_status.update_layout(title="Transaction Counts by Status", xaxis_title="Status", yaxis_title="Count")
st.plotly_chart(fig_status)

# Summary Report Visualization
st.subheader("Summary Report")
st.write(f"Total Transactions: {len(df_transactions)}")
st.write(f"Transactions Running: {df_transactions[df_transactions['Status'] == 'Running'].shape[0]}")
st.write(f"Transactions Waiting: {df_transactions[df_transactions['Status'] == 'Waiting'].shape[0]}")
st.write(f"Detected Deadlocks: {len(deadlocks)}")
