
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
from st_aggrid import AgGrid, GridOptionsBuilder

import matplotlib.pyplot as plt
st.set_page_config(page_title="Anomaly Detector", layout="wide")

st.title("🔍 First Timers Reconciliation Anomaly Catcher")

# --- Load or Train Model and compute SD values ---
@st.cache_resource
def train_model_and_stats():
    file_path = os.path.join(os.path.dirname(__file__), "HistoricalTrainingSet.xlsx")
    df = pd.read_excel(file_path, header=0)
    df['Balance Difference'] = pd.to_numeric(df['Balance Difference'], errors='coerce')
    df = df.dropna(subset=['Balance Difference'])

    df['Anomaly'] = df['Match Status'].apply(lambda x: 'Yes' if str(x).strip().lower() == 'break' else 'No')
    df['Anomaly_Label'] = df['Anomaly'].map({'Yes': 1, 'No': 0})

    X = df[['Balance Difference']]
    y = df['Anomaly_Label']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    mean_diff = df['Balance Difference'].mean()
    std_diff = df['Balance Difference'].std()

    return model, mean_diff, std_diff

model, mean_diff, std_diff = train_model_and_stats()

st.sidebar.markdown("### 🔧 Threshold Configuration")

std_options = {
    "Manual": None,
    "±1 SD": std_diff * 1,
    "±2 SD": std_diff * 2,
    "±3 SD": std_diff * 3,
}

std_choice = st.sidebar.radio("Select Deviation Threshold", options=list(std_options.keys()), index=0)

if std_choice != "Manual":
    threshold = round(std_options[std_choice], 2)
    st.sidebar.number_input("Current Threshold (Auto)", value=threshold, step=1000.0, key="auto_thresh", disabled=True)
else:
    threshold = st.sidebar.number_input("Enter Manual Threshold", value=10000.0, step=1000.0, key="manual_thresh")



st.sidebar.markdown("""🔖 **Anomaly Label Meaning**  
- 🟥 **High Deviation**: Balance Difference exceeds the selected threshold.  
- 🟦 **Zero Difference**: Difference between balances is exactly zero.  
- 🟩 **Manual Break**: Record marked as 'Break' in Match Status column.  
""")
uploaded_file = None
tabs = st.tabs(["📤 Upload File", "🔎 Anomalies", "📊 Visualizations"])

with tabs[0]:
    uploaded_file = st.file_uploader("Upload your reconciliation file", type=[".xlsx", ".csv"])

if uploaded_file:
    with tabs[1]:
        if uploaded_file.name.endswith(".csv"):
            new_df = pd.read_csv(uploaded_file)
        else:
            new_df = pd.read_excel(uploaded_file, header=0)

        new_df['Balance Difference'] = pd.to_numeric(new_df['Balance Difference'], errors='coerce')
        new_df = new_df.dropna(subset=['Balance Difference'])

        X_new = new_df[['Balance Difference']]
        new_df['Predicted Anomaly'] = model.predict(X_new)
        new_df['Predicted Anomaly'] = new_df['Predicted Anomaly'].map({1: 'Yes', 0: 'No'})

        def get_labels(row):
            labels = []
            diff = abs(row['Balance Difference'])
            if diff > threshold:
                labels.append("High Deviation")
            if diff == 0:
                labels.append("Zero Difference")
            if row.get('Match Status', '').strip().lower() == 'break':
                labels.append("Manual Break")
            return ", ".join(labels)

        new_df['Anomaly Labels'] = new_df.apply(lambda row: get_labels(row) if row['Predicted Anomaly'] == 'Yes' else "", axis=1)

        st.success("✅ Analysis Complete")
        
        def highlight_labels(val):
            if isinstance(val, str):
                style = []
                if 'High Deviation' in val:
                    style.append('background-color: #FFA07A')
                if 'Zero Difference' in val:
                    style.append('background-color: #ADD8E6')
                if 'Manual Break' in val:
                    style.append('background-color: #90EE90')
                return '; '.join(style)
            return ''

        
        def highlight_labels(val):
            if isinstance(val, str):
                style = []
                if 'High Deviation' in val:
                    style.append('color: #FF4500')  # orange-red
                if 'Zero Difference' in val:
                    style.append('color: #1E90FF')  # dodger blue
                if 'Manual Break' in val:
                    style.append('color: #228B22')  # forest green
                return '; '.join(style)
            return ''
        
        def colorize_multilabels(val):
            if isinstance(val, str) and val.strip():
                parts = val.split(', ')
                colored_parts = []
                for part in parts:
                    if part == 'High Deviation':
                        colored_parts.append(f'<span style="color:#FF4500">{part}</span>')
                    elif part == 'Zero Difference':
                        colored_parts.append(f'<span style="color:#1E90FF">{part}</span>')
                    elif part == 'Manual Break':
                        colored_parts.append(f'<span style="color:#228B22">{part}</span>')
                    else:
                        colored_parts.append(part)
                return ', '.join(colored_parts)
            return val

        
        gb = GridOptionsBuilder.from_dataframe(new_df)
        gb.configure_default_column(filter=True, sortable=True, resizable=True)
        gridOptions = gb.build()
        AgGrid(new_df, gridOptions=gridOptions, height=500, theme='streamlit')





        csv = new_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", data=csv, file_name="anomaly_results.csv", mime="text/csv")

        # Export with Excel styling
        styled_excel_path = "styled_anomalies.xlsx"

        def apply_excel_style(val):
            if isinstance(val, str) and 'High Deviation' in val:
                return 'background-color: #FFA07A'
            elif isinstance(val, str) and 'Zero Difference' in val:
                return 'background-color: #ADD8E6'
            elif isinstance(val, str) and 'Manual Break' in val:
                return 'background-color: #90EE90'
            return ''

        styled_df = new_df.style.applymap(apply_excel_style, subset=['Anomaly Labels'])
        styled_df.to_excel(styled_excel_path, index=False, engine='openpyxl')

        with open(styled_excel_path, 'rb') as f:
            st.download_button(
                label="📥 Download Styled Excel",
                data=f,
                file_name="styled_anomalies.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with tabs[2]:
        # Bar chart of anomalies grouped by labels
        label_colors = {
            "High Deviation": "#FFA07A",
            "Zero Difference": "#ADD8E6",
            "Manual Break": "#90EE90"
        }

        label_counts = new_df[new_df['Predicted Anomaly'] == 'Yes']['Anomaly Labels'].str.get_dummies(sep=", ").sum()
        fig, ax = plt.subplots()
        colors = [label_colors.get(label, '#D3D3D3') for label in label_counts.index]
        ax.bar(label_counts.index, label_counts.values, color=colors)
        ax.set_title("Anomalies Grouped by Labels")
        ax.set_ylabel("Count")
        ax.set_xticks(range(len(label_counts.index)))
        ax.set_xticklabels(label_counts.index, rotation=45, ha='right')
        st.pyplot(fig)
else:
    st.info("Upload a file to begin analysis.")
