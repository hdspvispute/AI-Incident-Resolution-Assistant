import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import plotly.express as px
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI as PandasAI_OpenAI
from PIL import Image
import os
import matplotlib
import sys

# 💡 Prevent opening PNG files externally on Windows
matplotlib.use("Agg")
if sys.platform.startswith("win"):
    os.startfile = lambda *args, **kwargs: None
# 🔐 Set your API key here

OPENAI_API_KEY="your-openai-key"

client = OpenAI(api_key=OPENAI_API_KEY)
pandasai_llm = PandasAI_OpenAI(api_token=OPENAI_API_KEY)

@st.cache_data(show_spinner=False)
def generate_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts)

def generate_dashboard(df):
    st.header("📊 Executive Incident Dashboard")

    if "Category" in df.columns:
        cat_count = df["Category"].value_counts().reset_index()
        cat_count.columns = ["Category", "Count"]
        st.plotly_chart(px.bar(cat_count, x="Category", y="Count", title="Incident Count by Category"))

    if "Assignment group" in df.columns:
        ag_count = df["Assignment group"].value_counts().reset_index()
        ag_count.columns = ["Assignment Group", "Count"]
        st.plotly_chart(px.bar(ag_count, x="Assignment Group", y="Count", title="Incidents by Assignment Group"))

    if "Made SLA" in df.columns:
        sla_count = df["Made SLA"].value_counts().reset_index()
        sla_count.columns = ["SLA Met", "Count"]
        st.plotly_chart(px.pie(sla_count, names="SLA Met", values="Count", title="SLA Compliance"))

    if "Reopen count" in df.columns:
        df["Reopen count"] = pd.to_numeric(df["Reopen count"], errors="coerce")
        st.plotly_chart(px.histogram(df, x="Reopen count", title="Reopen Frequency"))

    st.subheader("📋 Full Dataset")
    st.dataframe(df)

def main():
    st.title("🧠 AI Incident Resolution Assistant")

    uploaded_file = st.file_uploader("📂 Upload Incident Excel File", type=["xlsx"])
    row_limit = st.sidebar.number_input("🔢 Row limit", min_value=100, max_value=10000, value=500, step=100)

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            df = df.head(row_limit)
            st.success(f"✅ File loaded with {len(df)} rows.")

            tabs = st.tabs(["🧠 AI Assistant", "📊 Dashboard", "📈 Data Analysis"])

            with tabs[0]:
                st.sidebar.markdown("### 🔧 Column Mapping")
                short_col = st.sidebar.selectbox("Short Description", df.columns, index=df.columns.get_loc("Short description"))
                desc_col = st.sidebar.selectbox("Detailed Description", df.columns, index=df.columns.get_loc("Description"))
                resolution_col = st.sidebar.selectbox("Resolution Notes", df.columns, index=df.columns.get_loc("Resolution notes"))

                if 'confirmed' not in st.session_state:
                    st.session_state['confirmed'] = False

                if st.sidebar.button("✅ Confirm Mapping"):
                    st.session_state['confirmed'] = True
                    st.success("✅ Mapping confirmed.")

                if st.session_state['confirmed']:
                    if 'embeddings' not in st.session_state:
                        st.info("🔄 Generating embeddings...")
                        df["text"] = (
                            df[short_col].fillna('') + " " +
                            df[desc_col].fillna('') + " " +
                            df["Category"].fillna('') + " " +
                            df["Service"].fillna('') + " " +
                            df["Priority"].fillna('')
                        )
                        embeddings = generate_embeddings(df["text"].tolist())
                        index = faiss.IndexFlatL2(embeddings.shape[1])
                        index.add(np.array(embeddings))
                        st.session_state['embeddings'] = embeddings
                        st.session_state['index'] = index
                        st.session_state['df'] = df
                        st.session_state['model'] = SentenceTransformer("all-MiniLM-L6-v2")
                        st.success("✅ Ready! Please enter your problem below.")

                    query = st.text_input("💬 Describe your problem:")
                    if query:
                        st.info("🔍 Searching similar incidents...")
                        query_embedding = st.session_state['model'].encode([query])
                        D, I = st.session_state['index'].search(np.array(query_embedding), k=3)
                        matched_df = st.session_state['df'].iloc[I[0]]

                        context = "\n\n".join(
                            f"Issue: {row[short_col]} | Category: {row.get('Category', '')} | Service: {row.get('Service', '')} | Priority: {row.get('Priority', '')}\nResolution: {row[resolution_col]}"
                            for _, row in matched_df.iterrows() if str(row[resolution_col]).strip()
                        ) or "No strong matches found."

                        prompt = f"""You are an expert IT support assistant. A user reported: "{query}"

Here are historical incidents and their resolutions:

{context}

Based on this, suggest the best possible resolution:"""

                        with st.spinner("🤖 Generating resolution..."):
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": prompt}]
                            )
                            ai_answer = response.choices[0].message.content

                        st.markdown("### ✅ Suggested Resolution")
                        st.success(ai_answer)

                        st.markdown("### 📋 Related Incidents")
                        st.dataframe(matched_df)

            with tabs[1]:
                generate_dashboard(df)

            with tabs[2]:
                st.subheader("📈 Ask a Question About the Data")
                user_query = st.text_area("Type your data question below:")

                if user_query:
                    sdf = SmartDataframe(df, config={"llm": pandasai_llm})
                    with st.spinner("🔍 Analyzing data..."):
                        answer = sdf.chat(user_query)
                    st.success("✅ Done!")

                    if (
                        answer and isinstance(answer, str)
                        and answer.endswith(".png")
                        and os.path.exists(answer)
                    ):
                        st.image(answer, caption="📊 Generated Chart", use_container_width=True)
                    #st.image(Image.open(answer), caption="📊 Generated Chart", use_container_width=True)
                    else:
                        st.success(answer)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

if __name__ == "__main__":
    main()
