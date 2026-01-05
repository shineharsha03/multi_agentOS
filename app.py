import streamlit as st
import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fastembed import TextEmbedding
from groq import Groq
from pypdf import PdfReader

# --- CONFIGURATION ---
st.set_page_config(page_title="Universal AI Architect", page_icon="🧠", layout="wide")

# 1. FORCE MANUAL LOGIN (Always ask)
# The .strip() removes hidden spaces automatically
GROQ_API_KEY = st.text_input("Enter Groq API Key:", type="password").strip()

# 2. INITIALIZE CLIENT (Do not cache this!)
groq_client = Groq(api_key=GROQ_API_KEY)

# --- LOAD HEAVY RESOURCES (Cache these only) ---
@st.cache_resource
def load_heavy_models():
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    client = QdrantClient(":memory:") # RAM Mode
    return embedding_model, client

embedding_model, client = load_heavy_models()
COLLECTION_NAME = "knowledge_base"

# --- SIDEBAR: MISSION CONTROL ---
with st.sidebar:
    st.header("🎛️ Agent Controller")
    
    agent_mode = st.selectbox(
        "Select AI Persona:",
        ["Medical Appeal Shark", "Wall Street Analyst", "SaaS Customer Support"]
    )
    
    st.divider()
    
    st.write(f"📂 Upload Data for **{agent_mode}**")
    uploaded_file = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
    
    if uploaded_file and st.button("Train Agent"):
        with st.spinner("Processing Data..."):
            text_data = []
            if uploaded_file.name.endswith(".pdf"):
                reader = PdfReader(uploaded_file)
                for page in reader.pages:
                    text_data.append(page.extract_text())
            else: 
                text_data = [uploaded_file.read().decode("utf-8")]

            full_text = "\n".join(text_data)
            chunks = [chunk for chunk in full_text.split('\n') if chunk.strip()]
            
            if client.collection_exists(COLLECTION_NAME):
                client.delete_collection(COLLECTION_NAME)
            
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

            vectors = list(embedding_model.embed(chunks))
            points = [PointStruct(id=i, vector=v.tolist(), payload={"text": text}) for i, (text, v) in enumerate(zip(chunks, vectors))]
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            
        st.success(f"✅ {agent_mode} is ready!")

# --- DYNAMIC PROMPT SYSTEM ---
if agent_mode == "Medical Appeal Shark":
    st.title("⚖️ AppealOS: Denial Crusher")
    input_placeholder = "Describe the denied claim..."
    base_prompt = "You are a Senior Medical Billing Advocate. Write an aggressive, formal appeal letter. Cite the uploaded policy explicitly. Demand payment."

elif agent_mode == "Wall Street Analyst":
    st.title("📈 MarketMind: Stock Analyst")
    input_placeholder = "Ask about the stock..."
    base_prompt = "You are a Hedge Fund Analyst. Analyze the report. Look for risks and 'Golden Crossover' signals. Give a clear BUY/SELL recommendation."

else: # SaaS Support
    st.title("🤝 SaaS-Hero: Retention Expert")
    input_placeholder = "Paste the angry customer email here..."
    base_prompt = "You are a Senior Customer Success Manager. De-escalate the angry customer. REFUSE the refund if policy says 'No Refunds'. Offer a 'VIP Training Session' instead."

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(input_placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        query_vector = list(embedding_model.embed([prompt]))[0].tolist()
        search_results = client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=5).points
        context_text = "\n".join([r.payload['text'] for r in search_results if r.score > 0.4])
    except:
        context_text = ""

    if not context_text:
        response = "⚠️ I have no data. Please upload a document first."
    else:
        system_prompt = f"{base_prompt}\n\n--- DATA CONTEXT ---\n{context_text}"
        
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
            )
            response = chat_completion.choices[0].message.content
        except Exception as e:
            response = f"Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)