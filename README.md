# 🛡️ Agentic AI DDoS Detection & Response System

An end-to-end, locally hosted cybersecurity application that combines **traditional Machine Learning** for high-speed threat detection with an **Agentic Generative AI workflow** to automate incident analysis and mitigation responses. 

This project demonstrates a modern approach to AI-driven Security Operations Centers (SOC) by routing detected anomalies through a LangGraph-orchestrated LLM pipeline powered by Retrieval-Augmented Generation (RAG).

---

## 🚀 Features
* **Hybrid AI Detection:** Uses a Scikit-Learn Random Forest model to rapidly classify network traffic (Normal vs. SYN Flood) before engaging resource-heavy LLMs.
* **Agentic Workflow (LangGraph):** A customized Directed Acyclic Graph (DAG) that conditionally routes traffic. If an attack is detected, it triggers severity classification, threat intel retrieval, and LLM analysis.
* **Local AI (Privacy-First):** Uses **Llama 3.2 (1B)** via Ollama to run locally without sending sensitive network logs to external APIs like OpenAI.
* **Threat Intel RAG:** Integrates a PostgreSQL (`pgvector`) database to retrieve relevant MITRE ATT&CK mitigation strategies using HuggingFace sentence embeddings.
* **Automated SOC Reporting:** The AI acts as a SOC Analyst, generating a human-readable 3-sentence incident report and 3 immediate firewall mitigation steps.
* **REST API & Persistence:** Built with Flask and SQLAlchemy to serve inference endpoints and persistently log all incidents.

---

## 🧠 Architecture Flow
1. **`POST /api/analyze`**: Receives incoming network telemetry payload.
2. **Feature Extraction Node**: Parses numerical features (duration, bytes, counts).
3. **ML Detection Node**: Random Forest model assigns a confidence score and classifies the threat.
4. **Conditional Router**: If Normal, the graph ends. If an Attack is detected, it proceeds to investigation.
5. **Threat Intel Retrieval (RAG)**: Embeds the attack type and fetches context from `pgvector`.
6. **LLM Analysis Node**: Llama 3.2 digests the raw data + RAG context to write a localized incident report.
7. **Response Recommendation Node**: Llama 3.2 prescribes actionable firewall rules.
8. **Storage**: The complete AI report is saved to PostgreSQL and returned via the API.

---

## 🛠️ Tech Stack
* **Backend:** Python, Flask, SQLAlchemy
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **AI / Orchestration:** LangGraph, LangChain, Ollama (Llama 3.2:1b), HuggingFace (`all-MiniLM-L6-v2`)
* **Database:** PostgreSQL with `pgvector` extension

---

## ⚙️ Local Setup & Installation

### 1. Prerequisites
* Python 3.11+
* Docker (for the vector database)
* Ollama (for local LLM inference)

### 2. Start PostgreSQL with `pgvector`
Run the database locally using Docker. We map it to port `5434` to avoid conflicts with default Postgres installations.
```bash
docker run --name pgvector-ddos \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=ddos_db \
  -p 5434:5432 \
  -d ankane/pgvector
```

### 3. Pull the Local LLM
Ensure Ollama is running in the background, then download the lightweight Llama 3.2 model:
```bash
ollama pull llama3.2:1b
```

### 4. Install Python Dependencies
Clone the repository and install the required packages:
```bash
pip install -r requirements.txt
```

### 5. Run the Server
```bash
python3 main.py
```
*The Flask API will start on `http://127.0.0.1:5001`.*

---

## 🧪 API Usage & Testing

### 1. Generate a Mock Attack Payload
```bash
curl -X POST http://127.0.0.1:5001/api/simulate
```

### 2. Trigger the Agentic AI Pipeline
Submit a high-traffic payload to see the ML model detect the anomaly and the AI generate an incident report.
```bash
curl -X POST http://127.0.0.1:5001/api/analyze \
     -H "Content-Type: application/json" \
     -d '{
           "duration": 0, 
           "src_bytes": 0, 
           "dst_bytes": 0, 
           "count": 512, 
           "srv_count": 512,
           "protocol_type": "tcp",
           "flag": "S0"
         }'
```

### 3. View Saved Incidents
```bash
curl -X GET http://127.0.0.1:5001/api/incidents
```

---

## 🔮 Future Improvements
* **Real-time Kafka Integration:** Ingest live network streams instead of REST API payloads.
* **Web Dashboard:** Build a React.js frontend to visualize the incident reports and threat confidence graphs.
* **Advanced RAG:** Populate the pgvector database with thousands of MITRE ATT&CK techniques.

---
