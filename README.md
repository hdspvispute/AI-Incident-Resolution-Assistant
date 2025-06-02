
# 🧠 Incident AI Assistant with Streamlit

A smart, AI-powered incident analysis assistant built with **Streamlit**, **OpenAI**, **PandasAI**, and **FAISS**, enabling users to:
- Resolve IT incidents using historical data + LLMs
- Ask natural language questions about the dataset
- View dashboards and auto-generated charts
- Upload any incident Excel file and auto-map columns

---

## 🔧 Features

### 🚀 AI Incident Assistant
- Uses FAISS to find similar past incidents
- Summarizes resolutions using OpenAI GPT-3.5
- Learns from short and detailed descriptions, categories, priorities, and services

### 📊 Executive Dashboard
- Auto-generated charts for:
  - Incidents by category
  - Assignment group distribution
  - SLA compliance
  - Reopen count histogram
- Includes full dataset preview

### 📈 Data Analysis (PandasAI)
- Ask free-form questions like:
  - “Trend of high-priority incidents over months”
  - “Which assignment group has the most reopened tickets?”
- AI responds with insights or plots directly in the app

---

## 📁 Folder Structure

```
incident_ai_assistant/
│
├── incident_ai_assistant.py       # Main Streamlit app
├── requirements.txt               # Python dependencies
├── exports/                       # Temp folder for PandasAI chart exports
└── README.md                      # This file
```

---

## ⚙️ Setup Instructions

### 1. 🐍 Create and activate a virtual environment
```bash
python -m venv incident-venv
# Activate
# Windows:
incident-venv\Scripts\activate
# macOS/Linux:
source incident-venv/bin/activate
```

### 2. 📦 Install dependencies
```bash
pip install -r requirements.txt
```

### 3. 🔑 Set your OpenAI API Key (in code or environment)
In `incident_ai_assistant.py`:
```python
OPENAI_API_KEY = "your-openai-key"
```

Or use environment variable:
```bash
export OPENAI_API_KEY="your-openai-key"
```

---

## ▶️ Running the App

```bash
streamlit run incident_ai_assistant.py
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## 📝 Example Use Cases

- “User unable to log into VPN from remote location”  
  → Suggest resolution from similar incidents

- “Show trend of Category = 'Network' over time”  
  → Auto-generated line chart

- “Which service failed most SLA agreements?”  
  → Insight with bar chart or table

---

## 🧼 Notes

- Supports `.xlsx` files with dynamic column mapping
- Automatically handles unknown column names via sidebar
- Temp images from charts are **not opened externally**
- Streamlit deprecation warnings handled via `use_container_width=True`
- Matplotlib auto-open override applied on Windows

---

## 📊 Requirements

```txt
streamlit
pandas
openai>=1.0.0
pandasai
sentence-transformers
faiss-cpu
plotly
Pillow
```

---

## 📬 Contact

For improvements, issues, or collaboration:
- Developer: **Prafulla Vispute**
