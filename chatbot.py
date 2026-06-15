"""
Pakistan Budget 2026-2027 RAG Chatbot
======================================
Retrieval strategy:
  1. Chat history check (instant answer for repeated questions)
  2. Multi-query generation (specific, broad, Urdu, mixed)
  3. Pinecone vector search per query variant
  4. Parent document retrieval (full context)
  5. LLM-based reranking
  6. Answer generation with guardrails
  7. Store Q&A in history
"""

import json
import os
import sys
import uuid

from cerebras.cloud.sdk import Cerebras
from pinecone import Pinecone

import config


# ──────────────────────────────────────────────────────────────────────────────
#  Initialization
# ──────────────────────────────────────────────────────────────────────────────

def init_services():
    """Initialize Pinecone client, LLM, and load parent store."""
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    idx = pc.Index(name=config.PINECONE_INDEX_NAME, host=config.PINECONE_HOST_URL)

    llm = Cerebras(api_key=config.CEREBRAS_API_KEY)

    parent_store = load_parent_store()
    chat_history = load_chat_history()

    return pc, idx, llm, parent_store, chat_history


def llm_call(llm: Cerebras, messages: list[dict], max_tokens: int = 2048) -> str:
    """Make a Cerebras LLM call and return the content."""
    try:
        response = llm.chat.completions.create(
            model=config.CEREBRAS_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        choice = response.choices[0]
        msg = choice.message
        content = msg.content
        
        # Handle case where content is None or empty (reasoning model used all tokens on thinking)
        if not content and hasattr(msg, 'reasoning') and msg.reasoning:
            reasoning = msg.reasoning
            import re
            # Remove markdown formatting from reasoning
            clean = re.sub(r'\*\*?|\*\s*', '', reasoning)
            
            # Look for answer sections
            patterns = [
                r'\*?\*?Formulate.*?Answer[^:]*:?\*?\*?\s*\n([\s\S]+)',
                r'\*?\*?Final.*?Answer[^:]*:?\*?\*?\s*\n([\s\S]+)',
                r'Therefore,?\s+([\s\S]+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, clean, re.IGNORECASE)
                if match:
                    answer = match.group(1).strip()
                    # Clean up and take first 500 chars
                    answer = re.sub(r'\n\s*\n', '\n', answer)
                    answer = re.sub(r'\*\s*', '', answer)
                    if len(answer) > 30:
                        return answer[:500]
            
            # Fallback: extract the last meaningful paragraph
            paragraphs = [p.strip() for p in clean.split('\n\n') if len(p.strip()) > 20]
            if paragraphs:
                return paragraphs[-1][:500]
        
        return content if content else ""
    except Exception as e:
        print(f"   [LLM Error: {e}]")
        return ""


def load_parent_store() -> dict:
    """Load parent documents from local JSON."""
    if not os.path.exists(config.PARENT_STORE_PATH):
        print(f"⚠️  Parent store not found at {config.PARENT_STORE_PATH}")
        print("   Run `python ingest.py` first!")
        sys.exit(1)
    with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
        store = json.load(f)
    print(f"📚 Loaded {len(store)} parent documents")
    return store


def load_chat_history() -> list[dict]:
    """Load previous chat history from disk."""
    if os.path.exists(config.CHAT_HISTORY_PATH):
        with open(config.CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
        return history
    return []


def save_chat_history(history: list[dict]):
    """Persist chat history to disk."""
    with open(config.CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
#  Embedding helper
# ──────────────────────────────────────────────────────────────────────────────

def embed_query(pc: Pinecone, text: str) -> list[float]:
    """Embed a single query using Pinecone inference."""
    result = pc.inference.embed(
        model=config.EMBEDDING_MODEL,
        inputs=[text],
        parameters={"input_type": "query", "truncate": "END"},
    )
    return result.data[0].values


# ──────────────────────────────────────────────────────────────────────────────
#  Chat History Search (repeated question detection)
# ──────────────────────────────────────────────────────────────────────────────

def check_history(idx, pc: Pinecone, question: str) -> str | None:
    """Search Pinecone qa_history namespace for a similar past question."""
    try:
        query_emb = embed_query(pc, question)
        results = idx.query(
            namespace=config.NS_HISTORY,
            vector=query_emb,
            top_k=1,
            include_metadata=True,
        )
        matches = results.get("matches", [])
        if matches and matches[0]["score"] >= config.HISTORY_SIMILARITY_THRESHOLD:
            answer = matches[0]["metadata"].get("answer", "")
            if answer:
                return answer
    except Exception:
        pass
    return None


def store_qa_in_history(idx, pc: Pinecone, question: str, answer: str):
    """Embed and store the Q&A pair in Pinecone history namespace."""
    # Don't store unhelpful answers
    skip_phrases = ["couldn't find", "don't have enough", "couldn't generate"]
    if any(phrase in answer.lower() for phrase in skip_phrases):
        return
    try:
        query_emb = embed_query(pc, question)
        idx.upsert(
            namespace=config.NS_HISTORY,
            vectors=[{
                "id": str(uuid.uuid4()),
                "values": query_emb,
                "metadata": {
                    "question": question[:500],
                    "answer": answer[:1500],
                    "type": "qa_pair",
                },
            }],
        )
    except Exception:
        pass  # Non-critical; don't block the chatbot


# ──────────────────────────────────────────────────────────────────────────────
#  Multi-Query Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_multi_queries(llm, question: str) -> list[str]:
    """Generate query variants. Uses programmatic approach for reliability."""
    import re
    queries = []
    
    # Extract key terms from the question
    keywords = re.findall(r'\b[A-Za-z]+\b', question.lower())
    keywords = [k for k in keywords if k not in {'what', 'is', 'the', 'for', 'and', 'of', 'in', 'to', 'a', 'how', 'which'}]
    
    # Query 1: Add budget-related keywords
    queries.append(f"Pakistan budget 2026-27 {' '.join(keywords)} expenditure revenue allocation")
    
    # Query 2: Urdu translation (common budget terms)
    urdu_map = {
        'budget': 'بجٹ',
        'total': 'کل',
        'outlay': 'اخراجات',
        'expenditure': 'اخراجات',
        'revenue': 'آمدنی',
        'tax': 'ٹیکس',
        'deficit': ' خسارہ',
        'gdp': 'جی ڈی پی',
        'billion': 'ارب',
        'million': 'ملین',
        'pakistan': 'پاکستان',
    }
    urdu_keywords = []
    for k in keywords[:5]:
        if k in urdu_map:
            urdu_keywords.append(urdu_map[k])
        else:
            urdu_keywords.append(k)
    queries.append(f"پاکستان بجٹ 2026-27 {' '.join(urdu_keywords)}")
    
    # Query 3: Mixed English/Urdu
    mixed = f"Pakistan budget {' '.join(keywords[:3])} {' '.join(urdu_keywords[:2])}"
    queries.append(mixed)
    
    return queries


# ──────────────────────────────────────────────────────────────────────────────
#  Pinecone Retrieval (multi-query)
# ──────────────────────────────────────────────────────────────────────────────

def retrieve_multi_query(idx, pc: Pinecone, queries: list[str]) -> dict:
    """
    Search Pinecone for each query variant and collect unique parent results.
    Returns: {parent_id: {"score": float, "child_texts": [str], "metadata": dict}}
    """
    parent_results = {}

    for query in queries:
        try:
            query_emb = embed_query(pc, query)
            results = idx.query(
                namespace=config.NS_DOCS,
                vector=query_emb,
                top_k=config.TOP_K_PER_QUERY,
                include_metadata=True,
            )
            for match in results.get("matches", []):
                pid = match["metadata"].get("parent_id", "")
                if not pid:
                    continue
                if pid not in parent_results:
                    parent_results[pid] = {
                        "score": 0.0,
                        "child_texts": [],
                        "metadata": match["metadata"],
                    }
                parent_results[pid]["score"] = max(
                    parent_results[pid]["score"], match["score"]
                )
                preview = match["metadata"].get("text_preview", "")
                if preview:
                    parent_results[pid]["child_texts"].append(preview)
        except Exception:
            continue

    return parent_results


# ──────────────────────────────────────────────────────────────────────────────
#  Parent Document Retrieval
# ──────────────────────────────────────────────────────────────────────────────

def get_parent_documents(parent_results: dict, parent_store: dict) -> list[dict]:
    """Fetch full parent document content for matched results."""
    documents = []
    for pid, info in parent_results.items():
        if pid in parent_store:
            parent = parent_store[pid]
            documents.append({
                "parent_id": pid,
                "content": parent["content"],
                "score": info["score"],
                "source": parent["metadata"].get("source", "unknown"),
                "language": parent["metadata"].get("language", "unknown"),
                "page": parent["metadata"].get("page", 0),
            })
    # Sort by score descending
    documents.sort(key=lambda x: x["score"], reverse=True)
    return documents


# ──────────────────────────────────────────────────────────────────────────────
#  Reranking (score-based)
# ──────────────────────────────────────────────────────────────────────────────

def rerank_with_llm(llm, question: str, documents: list[dict]) -> list[dict]:
    """Rerank documents using Pinecone scores (LLM reranking unreliable with reasoning models)."""
    if not documents:
        return []
    # Simply sort by score and return top 5
    documents.sort(key=lambda x: x["score"], reverse=True)
    return documents[:5]


# ──────────────────────────────────────────────────────────────────────────────
#  Answer Generation
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Pakistan Budget 2026-2027 expert assistant.
Answer based ONLY on the provided context.
- Include specific numbers and figures
- Cite the source document
- If information is not in the context, say "This information is not available in the provided documents."
- Be concise - no explanations or reasoning
- Answer in the same language as the question"""

ANSWER_PROMPT = """Context from budget documents:
{context}

Question: {question}

Provide a direct answer with specific numbers. Do not explain your reasoning. If the answer is not in the context, just say so."""


def generate_answer(llm, question: str, context_docs: list[dict]) -> str:
    """Generate a final answer using the reranked context."""
    if not context_docs:
        return "I couldn't find relevant information in the budget documents to answer your question."

    # Build context from top reranked documents (keep concise)
    context_parts = []
    for i, doc in enumerate(context_docs[:3]):
        source = doc.get("source", "unknown")
        lang = doc.get("language", "unknown")
        content = doc["content"][:1200]
        context_parts.append(f"--- Document {i+1} ({source}, {lang}) ---\n{content}")

    context = "\n\n".join(context_parts)
    prompt = ANSWER_PROMPT.format(context=context, question=question)

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        text = llm_call(llm, messages)
        return text.strip() if text else "No response generated."
    except Exception as e:
        return f"Error generating answer: {e}"


# ──────────────────────────────────────────────────────────────────────────────
#  Guardrails
# ──────────────────────────────────────────────────────────────────────────────

def input_guardrail(llm, question: str) -> bool:
    """Check if the question is on-topic using keyword matching."""
    budget_keywords = [
        'budget', 'tax', 'revenue', 'expenditure', 'spending', 'deficit', 'surplus',
        'gdp', 'growth', 'inflation', 'fiscal', 'pension', 'salary', 'allocation',
        'development', 'psdp', 'federal', 'province', 'finance', 'economy', 'economic',
        'بجٹ', 'ٹیکس', 'آمدنی', 'اخراجات', 'خسارہ', 'پاکستان',
        'million', 'billion', 'rupees', 'percent', '2026', '2027'
    ]
    q_lower = question.lower()
    return any(kw in q_lower for kw in budget_keywords)


def output_guardrail(llm, question: str, answer: str) -> bool:
    """Check if the answer is quality - simple heuristic check."""
    # Allow all non-empty answers
    if not answer or len(answer.strip()) < 10:
        return False
    # Reject if it's just an error message
    if "error" in answer.lower() and "generating" in answer.lower():
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
#  Main Chat Loop
# ──────────────────────────────────────────────────────────────────────────────

def process_question(pc, idx, llm, parent_store, chat_history, question):
    """Full RAG pipeline for a single question."""
    print(f"\n{'─'*60}")
    print(f"🔍 Processing: {question}\n")

    # ── Step 1: Input guardrail ──────────────────────────────────────────
    if not input_guardrail(llm, question):
        answer = "Sorry, I can only answer questions related to Pakistan's Budget 2026-2027."
        print(f"🛡️  Input blocked by guardrail")
        return answer, chat_history

    # ── Step 2: Check chat history for similar questions ─────────────────
    cached = check_history(idx, pc, question)
    if cached:
        print("⚡ Found similar question in history – returning cached answer")
        chat_history.append({"question": question, "answer": cached, "cached": True})
        save_chat_history(chat_history)
        return cached, chat_history

    # ── Step 3: Generate multi-query variants ────────────────────────────
    print("📝 Generating query variants...")
    queries = generate_multi_queries(llm, question)
    queries.insert(0, question)  # Include original query
    for i, q in enumerate(queries):
        print(f"   Q{i}: {q[:80]}...")

    # ── Step 4: Multi-query retrieval from Pinecone ──────────────────────
    print("\n🔎 Retrieving from Pinecone...")
    parent_results = retrieve_multi_query(idx, pc, queries)
    print(f"   Found {len(parent_results)} unique parent documents")

    # ── Step 5: Fetch full parent documents ──────────────────────────────
    documents = get_parent_documents(parent_results, parent_store)
    if not documents:
        answer = "I couldn't find relevant information in the budget documents."
        chat_history.append({"question": question, "answer": answer})
        save_chat_history(chat_history)
        return answer, chat_history

    print(f"   Retrieved {len(documents)} parent documents")
    for doc in documents[:3]:
        print(f"   📄 {doc['source']} (p.{doc['page']}, {doc['language']}) – score: {doc['score']:.3f}")

    # ── Step 6: LLM reranking ───────────────────────────────────────────
    print("\n🏆 Reranking with LLM...")
    reranked = rerank_with_llm(llm, question, documents)
    print(f"   Top {len(reranked)} documents after reranking")

    # ── Step 7: Generate answer ──────────────────────────────────────────
    print("\n💬 Generating answer...")
    answer = generate_answer(llm, question, reranked)

    # ── Step 8: Output guardrail ─────────────────────────────────────────
    if not output_guardrail(llm, question, answer):
        answer = "I found relevant budget information but couldn't generate a complete response. Please try rephrasing your question."
        print("⚠️  Output did not meet quality standards")

    # ── Step 9: Store in history ─────────────────────────────────────────
    chat_history.append({"question": question, "answer": answer})
    save_chat_history(chat_history)
    store_qa_in_history(idx, pc, question, answer)

    return answer, chat_history


def main():
    print("=" * 60)
    print("  Pakistan Budget 2026-2027 – RAG Chatbot")
    print("  Type your question or '/quit' to exit")
    print("  Commands: /history, /clear, /stats")
    print("=" * 60)

    pc, idx, llm, parent_store, chat_history = init_services()

    while True:
        try:
            question = input("\n❓ You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue

        if question.lower() in ("/quit", "/exit", "/q"):
            print("\n👋 Goodbye!")
            break

        if question.lower() == "/history":
            if chat_history:
                for i, entry in enumerate(chat_history[-10:], 1):
                    q = entry["question"][:60]
                    print(f"  {i}. Q: {q}...")
            else:
                print("  No history yet.")
            continue

        if question.lower() == "/clear":
            chat_history = []
            save_chat_history([])
            try:
                idx.delete(delete_all=True, namespace=config.NS_HISTORY)
                print("  🗑️  History cleared (local + Pinecone).")
            except Exception:
                print("  🗑️  History cleared (local).")
            continue

        if question.lower() == "/stats":
            try:
                stats = idx.describe_index_stats()
                total = stats.get("total_vector_count", 0)
                ns = stats.get("namespaces", {})
                print(f"  📊 Total vectors: {total}")
                for name, info in ns.items():
                    print(f"     {name}: {info.get('record_count', 0)} vectors")
            except Exception as e:
                print(f"  Error fetching stats: {e}")
            continue

        try:
            answer, chat_history = process_question(
                pc, idx, llm, parent_store, chat_history, question
            )
            print(f"\n{'─'*60}")
            print(f"🤖 Bot: {answer}")
            print(f"{'─'*60}")
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
