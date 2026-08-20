"""
rag_layer.py
------------
RAG (Retrieval-Augmented Generation) Guidance Layer.
Queries ChromaDB for relevant wellness knowledge, then uses Groq LLM
to generate grounded, source-cited advice.
"""

import os
import sys
import json
import logging

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from openai import OpenAI
import chromadb

from ml.ml2.schemas import TriageVerdict, GuidanceResult
from ml.ml2.prompts import RAG_GUIDANCE_SYSTEM_PROMPT

# Load environment variables
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "wellness_knowledge"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
RAG_TOP_K = 3  # Number of chunks to retrieve

logger = logging.getLogger("ml2.rag_layer")

# Singleton ChromaDB collection
_CHROMA_COLLECTION = None


def _get_chroma_collection():
    """Singleton accessor for the ChromaDB collection."""
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None:
        if not os.path.exists(CHROMA_DB_DIR):
            raise FileNotFoundError(
                f"ChromaDB not found at {CHROMA_DB_DIR}. "
                "Run 'python -m ml.ml2.ingest_knowledge' first to load the knowledge base."
            )
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        _CHROMA_COLLECTION = client.get_collection(name=COLLECTION_NAME)
        logger.info(f"ChromaDB collection '{COLLECTION_NAME}' loaded ({_CHROMA_COLLECTION.count()} chunks)")
    return _CHROMA_COLLECTION


def _retrieve_relevant_chunks(verdict: TriageVerdict, summary: str, top_k: int = RAG_TOP_K) -> tuple[list[str], list[str]]:
    """
    Queries ChromaDB for chunks most relevant to the triage verdict.

    Returns:
        Tuple of (list of document texts, list of source filenames)
    """
    collection = _get_chroma_collection()

    # Build a natural language query from the verdict and summary
    query = f"Wellness guidance for condition: {verdict.value}. Context: {summary}"

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    documents = results["documents"][0] if results["documents"] else []
    sources = []
    if results["metadatas"]:
        sources = list({m["source"] for m in results["metadatas"][0]})

    logger.info(f"Retrieved {len(documents)} chunks from sources: {sources}")
    return documents, sources


def _call_groq_for_guidance(context_chunks: list[str], verdict: str, summary: str) -> dict:
    """
    Calls Groq LLM with retrieved context to generate grounded guidance.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set.")

    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    # Build context from retrieved chunks
    context_text = "\n\n---\n\n".join(context_chunks)

    user_prompt = f"""## User's Condition
Verdict: {verdict}
Summary: {summary}

## Reference Documents (use ONLY these for your advice)
{context_text}

Generate actionable wellness guidance based strictly on the reference documents above.
Remember: respond ONLY with the JSON format specified in your instructions."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": RAG_GUIDANCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=512,
    )

    raw = response.choices[0].message.content.strip()
    logger.debug(f"RAG guidance raw response: {raw}")

    # Parse JSON (handle markdown fences and unescaped newlines)
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Standard JSON parse failed. Attempting robust fallback parsing.")
        import re
        # Attempt to extract advice and sources using regex if JSON is malformed
        advice_match = re.search(r'"advice"\s*:\s*"(.*?)"\s*,?\s*"sources"', text, re.DOTALL)
        sources_match = re.search(r'"sources"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        
        advice = advice_match.group(1).replace("\n", " ") if advice_match else "Unable to parse advice."
        
        sources = []
        if sources_match:
            # Extract items like "doc.md" from the array string
            sources = [s.strip().strip('"').strip("'") for s in sources_match.group(1).split(",") if s.strip()]
            
        return {"advice": advice, "sources": sources}


def generate_guidance(verdict: TriageVerdict, summary: str) -> GuidanceResult:
    """
    Main entry point for RAG Guidance.

    1. Retrieves relevant chunks from ChromaDB
    2. Sends them to Groq with the verdict for grounded advice generation
    3. Returns structured GuidanceResult

    Parameters:
        verdict (TriageVerdict): The triage verdict from the Judge Agent.
        summary (str): The Judge's plain-English summary.

    Returns:
        GuidanceResult: Advice + source citations + disclaimer.
    """
    logger.info(f"Generating RAG guidance for verdict: {verdict.value}")

    try:
        # Step 1: Retrieve relevant knowledge
        chunks, sources = _retrieve_relevant_chunks(verdict, summary)

        if not chunks:
            logger.warning("No relevant chunks found in knowledge base.")
            return GuidanceResult(
                advice="No specific guidance available for this condition. Please consult a healthcare professional if you have concerns.",
                sources=[],
            )

        # Step 2: Generate grounded guidance via Groq
        parsed = _call_groq_for_guidance(chunks, verdict.value, summary)

        return GuidanceResult(
            advice=parsed.get("advice", "No specific guidance available."),
            sources=parsed.get("sources", sources),
        )

    except FileNotFoundError as e:
        logger.warning(f"Knowledge base not found: {e}")
        return GuidanceResult(
            advice="Knowledge base not loaded. Run 'python -m ml.ml2.ingest_knowledge' to set up the RAG layer.",
            sources=[],
        )
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse RAG response: {e}")
        return GuidanceResult(
            advice="Unable to generate guidance at this time. Please try again.",
            sources=[],
        )
    except Exception as e:
        logger.error(f"RAG guidance generation error: {e}")
        return GuidanceResult(
            advice="An error occurred while generating guidance. Please try again.",
            sources=[],
        )


# -----------------------------------------------------------------------
# Standalone Test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    print("=" * 60)
    print("  ML-2 RAG Guidance Layer — Test")
    print("=" * 60)

    # Test with different verdicts
    test_cases = [
        (TriageVerdict.HIGH_STRESS, "You appear to be experiencing elevated stress based on high heart rate and low HRV."),
        (TriageVerdict.DROWSINESS, "Signs of drowsiness detected including frequent eye closures and head drooping."),
        (TriageVerdict.NORMAL, "Your vital signs and behavioral indicators appear within normal ranges."),
    ]

    for verdict, summary in test_cases:
        print(f"\n--- Verdict: {verdict.value} ---")
        result = generate_guidance(verdict, summary)
        print(f"Advice:\n{result.advice}")
        print(f"Sources: {result.sources}")
        print(f"Disclaimer: {result.disclaimer}")

    print("\n[DONE] RAG layer test complete!")
