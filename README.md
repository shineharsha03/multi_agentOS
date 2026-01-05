# 🧠 Universal AI Architect: Multi-Agent RAG System

### **Enterprise-Grade AI with Dynamic Context Switching**

**Live Demo:** [Link to your Streamlit App](https://share.streamlit.io) *(Paste your link here later)*

---

## 🚀 Project Overview
This application is a **Multi-Agent RAG (Retrieval-Augmented Generation)** system designed to solve complex business problems across distinct industries. 

Unlike standard chatbots, this architecture separates **Knowledge** (uploaded documents) from **Reasoning** (the AI Persona). By dynamically swapping system prompts and vector collections, a single codebase serves three distinct enterprise roles:

| Agent Persona | Domain | Business Logic |
| :--- | :--- | :--- |
| **⚖️ Medical Appeal Shark** | Healthcare | Analyzing insurance denials, cross-referencing policy PDF exclusions, and drafting strict legal appeals. |
| **📈 Wall Street Analyst** | Finance | Processing financial reports to identify "Golden Crossover" technical signals and assess market risk. |
| **🤝 SaaS Retention Lead** | Customer Success | detecting sentiment in angry support tickets and enforcing "No Refund" policies while maintaining high empathy (Churn Prevention). |

---

## 🛠️ Tech Stack & Architecture

* **Frontend:** Streamlit (Python)
* **LLM Engine:** Llama-3 (via Groq API) for <1s inference latency.
* **Vector Database:** Qdrant (In-Memory) for high-speed semantic search.
* **Embeddings:** FastEmbed (BAAI/bge-small-en-v1.5).
* **Orchestration:** Custom Python Logic (No heavy frameworks to reduce overhead).

### **How It Works**
1.  **Ingestion:** User uploads a PDF (Policy, Financial Report, or Email).
2.  **Vectorization:** Text is chunked and embedded into a high-dimensional vector space.
3.  **retrieval:** When a user queries, the system performs a Cosine Similarity search to find the top 5 relevant chunks.
4.  **Augmentation:** The specific "Agent Persona" prompt is combined with the retrieved data to generate a context-aware response.

---

## 💻 Installation

To run this locally:

```bash
# 1. Clone the repo
git clone [https://github.com/YOUR_USERNAME/multi_agentOS.git](https://github.com/YOUR_USERNAME/multi_agentOS.git)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py