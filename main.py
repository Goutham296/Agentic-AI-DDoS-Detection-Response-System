import os
import json
import uuid
import time
import logging
from datetime import datetime
from typing import TypedDict, Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

# LangChain / LangGraph Imports
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIG & SETUP
# ==========================================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:password@localhost:5434/ddos_db")
app.config["SQLALCHEMY_DATABASE_URI"] = DB_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ==========================================
# 2. DATABASE MODELS (PostgreSQL)
# ==========================================
class Incident(db.Model):
    __tablename__ = "incidents"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    detected_at = db.Column(db.DateTime, default=datetime.utcnow)
    attack_type = db.Column(db.String(100))
    severity = db.Column(db.String(20))
    confidence_score = db.Column(db.Float)
    ml_prediction = db.Column(db.JSON)
    llm_analysis = db.Column(db.Text)
    mitigation_steps = db.Column(db.Text)
    raw_features = db.Column(db.JSON)

with app.app_context():
    try:
        db.create_all()
    except Exception as e:
        log.warning(f"Database connection failed, running without DB: {e}")

# ==========================================
# 3. MACHINE LEARNING MODEL (NSL-KDD)
# ==========================================
MODEL_PATH = "models/ddos_rf_model_v2.pkl"

def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    
    log.info("Model not found. Initializing Fallback Random Forest...")
    # For a production scenario, you would train on the actual KDDTrain+.txt dataset
    # Here we instantiate a dummy untrained model or rely on simulation for the demo
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Create dummy training data:
    # Normal traffic (low count/srv_count) -> label 0
    # Attack traffic (high count/srv_count) -> label 1
    X_dummy = np.array([
        [0, 150, 200, 5, 5],
        [0, 200, 300, 10, 10],
        [0, 50, 100, 2, 2],
        [0, 0, 0, 500, 500],
        [0, 0, 0, 512, 512],
        [0, 0, 0, 600, 600]
    ])
    y_dummy = np.array([0, 0, 0, 1, 1, 1])
    rf.fit(X_dummy, y_dummy)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    return rf

classifier = load_or_train_model()

# ==========================================
# 4. RAG SETUP (pgVector)
# ==========================================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    vector_store = PGVector(
        connection_string=DB_URL,
        embedding_function=embeddings,
        collection_name="mitre_ddos_intel"
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    log.warning(f"PGVector setup failed (requires pgvector extension): {e}")
    retriever = None

# ==========================================
# 5. LANGGRAPH DAG: STATE & NODES
# ==========================================
class AgentState(TypedDict):
    raw_traffic_data: dict
    extracted_features: Optional[np.ndarray]
    attack_detected: bool
    attack_type: Optional[str]
    confidence_score: Optional[float]
    severity: Optional[str]
    threat_context: Optional[str]
    llm_analysis: Optional[str]
    mitigation_steps: Optional[str]

def feature_extraction_node(state: AgentState):
    """Extracts required numerical features from the raw network payload."""
    log.info("LangGraph: Extracting network features...")
    raw_data = state["raw_traffic_data"]
    
    # Expected features based on NSL-KDD base processing
    features = np.array([[
        float(raw_data.get("duration", 0)),
        float(raw_data.get("src_bytes", 0)),
        float(raw_data.get("dst_bytes", 0)),
        float(raw_data.get("count", 0)),
        float(raw_data.get("srv_count", 0))
    ]])
    return {"extracted_features": features}

def ml_detection_node(state: AgentState):
    """Runs the Machine Learning model to detect attacks."""
    log.info("LangGraph: Running ML classification...")
    features = state["extracted_features"]
    
    try:
        # Predict using loaded model
        pred = classifier.predict(features)[0]
        prob = np.max(classifier.predict_proba(features)[0])
        attack_detected = bool(pred == 1)
        attack_type = "DoS/SYN Flood" if attack_detected else "Normal"
    except Exception as e:
        # Fallback heuristic simulation if model fails
        log.warning(f"ML prediction failed, using heuristic fallback: {e}")
        attack_detected = state["raw_traffic_data"].get("count", 0) > 100
        attack_type = "DoS/SYN Flood" if attack_detected else "Normal"
        prob = 0.95 if attack_detected else 0.85
        
    return {
        "attack_detected": attack_detected, 
        "attack_type": attack_type, 
        "confidence_score": float(prob)
    }

def severity_classification_node(state: AgentState):
    """Classifies incident severity based on confidence and traffic volume."""
    log.info("LangGraph: Assigning severity score...")
    score = state.get("confidence_score", 0.0)
    count = state["raw_traffic_data"].get("count", 0)
    
    if score > 0.9 and count > 500:
        sev = "Critical"
    elif score > 0.8:
        sev = "High"
    elif score > 0.6:
        sev = "Medium"
    else:
        sev = "Low"
        
    return {"severity": sev}

def threat_intel_retrieval_node(state: AgentState):
    """RAG: Retrieves historical context and mitigation from pgVector."""
    log.info("LangGraph: Retrieving RAG threat intelligence...")
    if retriever:
        try:
            docs = retriever.invoke(f"DDoS Attack Type: {state['attack_type']}")
            context = "\n".join([d.page_content for d in docs])
        except Exception as e:
            log.error(f"Retriever error: {e}")
            context = "MITRE ATT&CK: SYN Flood involves sending rapid SYN packets to exhaust server resources."
    else:
        context = "MITRE ATT&CK: SYN Flood involves sending rapid SYN packets to exhaust server resources."
        
    return {"threat_context": context}

def llm_analysis_node(state: AgentState):
    """LLM Agent analyzes the incident using retrieved threat intel."""
    log.info("LangGraph: Generating LLM Incident Report (This may take 30-60 seconds locally)...")
    try:
        llm = ChatOllama(model="llama3.2:1b")
        prompt = ChatPromptTemplate.from_template(
            "You are an AI SOC Analyst. A network attack was detected.\n"
            "Attack Type: {attack_type}\n"
            "Severity: {severity}\n"
            "Confidence: {confidence_score}\n\n"
            "Threat Intel Context:\n{threat_context}\n\n"
            "Write a concise 3-sentence incident analysis report."
        )
        chain = prompt | llm | StrOutputParser()
        analysis = chain.invoke(state)
    except Exception as e:
        log.warning(f"LLM Analysis failed: {e}")
        analysis = f"Automated Alert: {state['attack_type']} detected with {state['severity']} severity."
        
    return {"llm_analysis": analysis}

def response_recommendation_node(state: AgentState):
    """Generates automated mitigation strategies via LLM."""
    log.info("LangGraph: Generating LLM Mitigation Steps (Almost done!)...")
    try:
        llm = ChatOllama(model="llama3.2:1b")
        prompt = ChatPromptTemplate.from_template(
            "Based on the following incident analysis:\n{llm_analysis}\n\n"
            "Provide 3 immediate firewall/networking mitigation steps in a numbered list."
        )
        chain = prompt | llm | StrOutputParser()
        mitigation = chain.invoke({"llm_analysis": state["llm_analysis"]})
    except Exception as e:
        mitigation = "1. Block source IP.\n2. Rate limit incoming traffic.\n3. Escalate to network admin."
        
    return {"mitigation_steps": mitigation}

# ==========================================
# 6. COMPILE DAG WORKFLOW
# ==========================================
def router(state: AgentState):
    # Only proceed to threat intel and LLM analysis if an attack is actually detected
    return "severity_classification_node" if state.get("attack_detected") else END

graph = StateGraph(AgentState)
graph.add_node("feature_extraction_node", feature_extraction_node)
graph.add_node("ml_detection_node", ml_detection_node)
graph.add_node("severity_classification_node", severity_classification_node)
graph.add_node("threat_intel_retrieval_node", threat_intel_retrieval_node)
graph.add_node("llm_analysis_node", llm_analysis_node)
graph.add_node("response_recommendation_node", response_recommendation_node)

graph.set_entry_point("feature_extraction_node")
graph.add_edge("feature_extraction_node", "ml_detection_node")

# Routing conditionally based on detection
graph.add_conditional_edges(
    "ml_detection_node", 
    router,
    {"severity_classification_node": "severity_classification_node", END: END}
)

graph.add_edge("severity_classification_node", "threat_intel_retrieval_node")
graph.add_edge("threat_intel_retrieval_node", "llm_analysis_node")
graph.add_edge("llm_analysis_node", "response_recommendation_node")
graph.add_edge("response_recommendation_node", END)

compiled_graph = graph.compile()

# ==========================================
# 7. FLASK REST APIs
# ==========================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "DDoS AI Agentic Backend is running! 🚀",
        "endpoints": [
            "POST /api/simulate",
            "POST /api/analyze",
            "GET /api/incidents"
        ]
    })

@app.route("/api/analyze", methods=["POST"])
def analyze_traffic():
    data = request.json
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
        
    initial_state = {"raw_traffic_data": data}
    
    # Run the LangGraph agentic pipeline
    final_state = compiled_graph.invoke(initial_state)
    
    # Clean non-serializable objects (like numpy arrays) for JSON response
    if "extracted_features" in final_state:
        del final_state["extracted_features"]
        
    if final_state.get("attack_detected"):
        try:
            # Store the incident in PostgreSQL
            incident = Incident(
                attack_type=final_state.get("attack_type"),
                severity=final_state.get("severity"),
                confidence_score=final_state.get("confidence_score"),
                ml_prediction={"prediction": final_state.get("attack_type")},
                llm_analysis=final_state.get("llm_analysis"),
                mitigation_steps=final_state.get("mitigation_steps"),
                raw_features=data
            )
            db.session.add(incident)
            db.session.commit()
        except Exception as e:
            log.error(f"Failed to save incident to DB: {e}")
            db.session.rollback()
        
    return jsonify({
        "status": "success", 
        "attack_detected": final_state.get("attack_detected"),
        "analysis": final_state
    })

@app.route("/api/incidents", methods=["GET"])
def get_incidents():
    try:
        incidents = Incident.query.order_by(Incident.detected_at.desc()).limit(50).all()
        return jsonify([{
            "id": inc.id,
            "detected_at": inc.detected_at,
            "attack_type": inc.attack_type,
            "severity": inc.severity,
            "confidence_score": inc.confidence_score,
            "llm_analysis": inc.llm_analysis,
            "mitigation_steps": inc.mitigation_steps
        } for inc in incidents])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/simulate", methods=["POST"])
def simulate_attack():
    """Endpoint for generating a sample attack payload to test the pipeline."""
    sample_attack = {
        "duration": 0,
        "src_bytes": 0,
        "dst_bytes": 0,
        "count": 512,  # high count simulates SYN flood
        "srv_count": 512,
        "protocol_type": "tcp",
        "flag": "S0"
    }
    return jsonify(sample_attack)

if __name__ == "__main__":
    log.info("Starting DDoS AI Agentic Backend on port 5001...")
    app.run(host="0.0.0.0", port=5001, debug=True)