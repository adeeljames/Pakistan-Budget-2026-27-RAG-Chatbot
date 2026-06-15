"""
Pakistan Budget 2026-2027 RAG Chatbot – Streamlit UI
=====================================================
A beautiful, modern chat interface for querying Pakistan's federal budget documents.
"""

import streamlit as st
import time
import json
import os

# Import from our chatbot module
from chatbot import (
    init_services,
    embed_query,
    check_history,
    store_qa_in_history,
    generate_multi_queries,
    retrieve_multi_query,
    get_parent_documents,
    rerank_with_llm,
    generate_answer,
    input_guardrail,
    output_guardrail,
)
import config

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pakistan Budget 2026-27 | AI Chatbot",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ── Base ─────────────────────────────────────────────── */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    }
    
    /* Hide default Streamlit header/footer */
    #MainMenu, header, footer {visibility: hidden;}
    
    /* ── Hero Header ──────────────────────────────────────── */
    .hero-header {
        background: linear-gradient(135deg, #00611c 0%, #01411C 30%, #00611c 60%, #ffffff 100%);
        padding: 2rem 2rem 1.5rem;
        border-radius: 0 0 24px 24px;
        margin: -1rem -1rem 1.5rem -1rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 97, 28, 0.3);
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        animation: shimmer 8s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        50% { transform: translate(5%, 5%) rotate(3deg); }
    }
    
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
        margin: 0;
        position: relative;
        z-index: 1;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.85);
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
        font-weight: 400;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.75rem;
        color: white;
        margin-top: 0.75rem;
        position: relative;
        z-index: 1;
        border: 1px solid rgba(255,255,255,0.15);
    }
    
    /* ── Chat Messages ────────────────────────────────────── */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .user-message {
        background: linear-gradient(135deg, #00611c, #017a24);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.75rem 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 4px 16px rgba(0, 97, 28, 0.25);
        animation: slideInRight 0.3s ease-out;
    }
    
    .bot-message {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        color: #e8e8e8;
        padding: 1.25rem 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.75rem 0;
        max-width: 85%;
        font-size: 0.95rem;
        line-height: 1.7;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 24px rgba(0,0,0,0.15);
        animation: slideInLeft 0.3s ease-out;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* ── Typing Indicator ─────────────────────────────────── */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 1rem 1.5rem;
        background: rgba(255,255,255,0.05);
        border-radius: 18px;
        max-width: 80px;
        margin: 0.5rem 0;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #00611c;
        border-radius: 50%;
        animation: typingBounce 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typingBounce {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-8px); }
    }
    
    /* ── Sidebar ──────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ccc;
    }
    
    .sidebar-section {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    
    .question-chip {
        display: block;
        background: rgba(0, 97, 28, 0.15);
        border: 1px solid rgba(0, 97, 28, 0.3);
        color: #8fd4a4;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        cursor: pointer;
        font-size: 0.85rem;
        transition: all 0.2s ease;
        text-decoration: none;
    }
    
    .question-chip:hover {
        background: rgba(0, 97, 28, 0.3);
        border-color: rgba(0, 97, 28, 0.5);
        transform: translateX(4px);
    }
    
    /* ── Input Area ───────────────────────────────────────── */
    .stChatInputContainer {
        background: transparent !important;
        border-top: 1px solid rgba(255,255,255,0.05) !important;
        padding-top: 1rem !important;
    }
    
    /* ── Footer ───────────────────────────────────────────── */
    .custom-footer {
        text-align: center;
        padding: 2rem 0 1rem;
        margin-top: 2rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    
    .footer-credit {
        color: rgba(255,255,255,0.5);
        font-size: 0.85rem;
        font-weight: 400;
    }
    
    .footer-credit a {
        color: #4ade80;
        text-decoration: none;
        font-weight: 600;
        transition: color 0.2s;
    }
    
    .footer-credit a:hover {
        color: #86efac;
        text-decoration: underline;
    }
    
    .footer-heart {
        color: #ef4444;
        animation: heartbeat 1.5s ease-in-out infinite;
        display: inline-block;
    }
    
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.2); }
    }
    
    /* ── Stat Cards ───────────────────────────────────────── */
    .stat-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4ade80;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }
    
    /* ── Welcome Card ─────────────────────────────────────── */
    .welcome-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 2rem auto;
        max-width: 600px;
    }
    
    .welcome-emoji {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .welcome-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e8e8e8;
        margin-bottom: 0.5rem;
    }
    
    .welcome-text {
        color: rgba(255,255,255,0.5);
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* ── Source tag ────────────────────────────────────────── */
    .source-tag {
        display: inline-block;
        background: rgba(0, 97, 28, 0.15);
        border: 1px solid rgba(0, 97, 28, 0.25);
        color: #8fd4a4;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.7rem;
        margin-top: 0.5rem;
    }
    
    /* ── Scrollbar ────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
    
</style>
""", unsafe_allow_html=True)


# ─── Sample Questions ──────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "What is the Federal PSDP allocation for 2026-27?",
    "What are the green revenue components?",
    "How much is allocated for development expenditure?",
    "What is the current expenditure budget?",
    "What are the grants and transfers for 2026-27?",
    "پاکستان بجٹ 2026-27 کا کل حجم کیا ہے؟",
    "What is the capital outlay on railways?",
    "What is the defence deposits and reserves allocation?",
]

# ─── Initialize Session State ─────────────────────────────────────────────────

def initialize_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "services" not in st.session_state:
        try:
            st.session_state.services = init_services()
            st.session_state.init_error = None
        except Exception as e:
            st.session_state.services = None
            st.session_state.init_error = str(e)
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0


def get_services():
    """Get initialized services."""
    if st.session_state.services is None:
        return None, None, None, None, None
    return st.session_state.services


# ─── Hero Header ───────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🇵🇰 Pakistan Budget 2026-27</div>
        <div class="hero-subtitle">AI-Powered RAG Chatbot · Ask anything about the federal budget</div>
        <div class="hero-badge">⚡ Powered by Pinecone · Cerebras GLM-4.7 · Multi-Query RAG</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### 📊 Budget Documents")
        
        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">3</div>
                <div class="stat-label">PDF Sources</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">461</div>
                <div class="stat-label">Vectors</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 💡 Try These Questions")
        st.markdown("*Click any question to ask:*")
        
        for i, q in enumerate(SAMPLE_QUESTIONS):
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📄 Source Documents")
        st.markdown("""
        - 📘 **Annual Budget Statement** (57 pages)
        - 📗 **Budget in Brief** (50 pages)  
        - 📕 **FM Speech Urdu** (53 pages)
        """)
        
        st.markdown("---")
        st.markdown("### 🔧 Pipeline")
        st.markdown("""
        1. Multi-Query (4 variants)
        2. Pinecone Vector Search
        3. Parent Document Retrieval
        4. Score-based Reranking
        5. Cerebras GLM-4.7 Answer
        6. Guardrails Check
        """)


# ─── Footer ────────────────────────────────────────────────────────────────────

def render_footer():
    st.markdown("""
    <div class="custom-footer">
        <div class="footer-credit">
            Made with <span class="footer-heart">❤️</span> by 
            <a href="https://www.linkedin.com/in/muhammadadeelai/" target="_blank">@muhammadadeelai</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Welcome Screen ────────────────────────────────────────────────────────────

def render_welcome():
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-emoji">🏛️</div>
        <div class="welcome-title">Ask About Pakistan's Budget 2026-27</div>
        <div class="welcome-text">
            I can answer questions about federal expenditure, revenue, PSDP allocations, 
            development budgets, tax targets, and more. Ask in English or Urdu!
            <br><br>
            <em>Try a sample question from the sidebar, or type your own below.</em>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Process Question (Streamlit Adapted) ──────────────────────────────────────

def process_question_streamlit(question: str):
    """Process a question and yield streaming updates."""
    pc, idx, llm, parent_store, chat_history = get_services()
    
    if pc is None:
        yield "error", "Services not initialized. Please refresh the page."
        return
    
    # Input guardrail
    if not input_guardrail(llm, question):
        yield "answer", "I can only answer questions related to Pakistan's Budget 2026-2027. Please ask about budget allocations, revenue, expenditure, PSDP, taxes, or fiscal policy."
        return
    
    # Check history
    yield "status", "Checking history..."
    cached = check_history(idx, pc, question)
    if cached:
        yield "status", "⚡ Found in history!"
        yield "answer", cached
        return
    
    # Multi-query
    yield "status", "🔍 Generating search queries..."
    queries = generate_multi_queries(llm, question)
    queries.insert(0, question)
    yield "debug", f"Queries: {len(queries)} variants generated"
    
    # Retrieval
    yield "status", "📚 Searching budget documents..."
    parent_results = retrieve_multi_query(idx, pc, queries)
    yield "debug", f"Found {len(parent_results)} relevant sections"
    
    # Parent docs
    documents = get_parent_documents(parent_results, parent_store)
    if not documents:
        yield "answer", "I couldn't find relevant information in the budget documents for this question."
        return
    
    # Reranking
    yield "status", "🏆 Reranking results..."
    reranked = rerank_with_llm(llm, question, documents)
    
    # Sources info
    sources = []
    for doc in reranked[:3]:
        sources.append(f"{doc['source']} (p.{doc['page']})")
    
    # Generate answer
    yield "status", "💬 Generating answer..."
    answer = generate_answer(llm, question, reranked)
    
    # Output guardrail
    if not output_guardrail(llm, question, answer):
        answer = "I found relevant information but couldn't generate a complete response. Please try rephrasing your question."
    
    # Add source attribution
    if sources and answer and "not available" not in answer.lower():
        source_text = " · ".join(set(sources))
        answer = f"{answer}\n\n📎 *Source: {source_text}*"
    
    # Store in history
    st.session_state.services = (pc, idx, llm, parent_store, 
                                  chat_history + [{"question": question, "answer": answer}])
    store_qa_in_history(idx, pc, question, answer)
    
    yield "answer", answer


# ─── Main App ──────────────────────────────────────────────────────────────────

def main():
    initialize_state()
    render_header()
    render_sidebar()
    
    # Check initialization
    if st.session_state.init_error:
        st.error(f"⚠️ Failed to initialize: {st.session_state.init_error}")
        st.info("Make sure you've run `python ingest.py` first and your .env file is configured.")
        render_footer()
        return
    
    # Handle pending question from sidebar
    pending = st.session_state.pop("pending_question", None)
    
    # Render chat messages
    if not st.session_state.messages:
        render_welcome()
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-message">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="bot-message">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
    
    # Chat input
    user_input = st.chat_input("Ask about Pakistan's Budget 2026-27...", key="chat_input")
    question = pending or user_input
    
    if question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Show user message immediately
        st.markdown(
            f'<div class="user-message">{question}</div>',
            unsafe_allow_html=True,
        )
        
        # Show typing indicator
        typing_placeholder = st.empty()
        typing_placeholder.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Process question
        status_placeholder = st.empty()
        answer = ""
        
        for event_type, event_data in process_question_streamlit(question):
            if event_type == "status":
                status_placeholder.markdown(
                    f'<div style="color: rgba(255,255,255,0.4); font-size: 0.8rem; padding: 0.25rem 0;">{event_data}</div>',
                    unsafe_allow_html=True,
                )
            elif event_type == "debug":
                pass  # Skip debug messages in UI
            elif event_type == "answer":
                answer = event_data
            elif event_type == "error":
                answer = f"❌ {event_data}"
        
        # Clear typing/status indicators
        typing_placeholder.empty()
        status_placeholder.empty()
        
        # Show answer
        if answer:
            # Convert markdown-style formatting to HTML
            formatted_answer = answer.replace("\n", "<br>")
            # Handle italic text *text*
            import re
            formatted_answer = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', formatted_answer)
            
            st.markdown(
                f'<div class="bot-message">{formatted_answer}</div>',
                unsafe_allow_html=True,
            )
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.question_count += 1
        
        # Rerun to show updated messages
        st.rerun()
    
    render_footer()


if __name__ == "__main__":
    main()
