"""RAG v1, the four stages with no external database."""

from typing import Any, Dict, List
import numpy as np

ABSTENTION_MESSAGE = (
    "The provided documents do not contain enough information to answer this."
)

GROUNDED_SYSTEM_PROMPT = """You are a document question answering assistant.

RULES
1. Answer using ONLY the numbered context blocks below.
2. Cite every claim with the block number in square brackets, for example [2].
3. If the context does not contain the answer, reply exactly:
   "The provided documents do not contain enough information to answer this."
4. Never use knowledge from outside the context blocks.
5. Keep the answer under 200 words unless the question asks for a list.
"""


### Document Loading
def load_document(file) -> str:
    """Parse an uploaded txt, md, or pdf file into text."""
    filename = file.name.lower()

    if filename.endswith(".pdf"):
        from pypdf import PdfReader

        text = ""
        for page in PdfReader(file).pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    if filename.endswith((".txt", ".md")):
        return file.getvalue().decode("utf-8")

    raise ValueError("Unsupported file type. Upload a txt, md, or pdf file.")


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Naive fixed size splitting, the baseline every later strategy must beat."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must exceed overlap")

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


### Indexing and Retrieval
def build_simple_index(chunks: List[str], embeddings_model) -> List[Dict[str, Any]]:
    """Embed every chunk once and hold the vectors in process memory."""
    index: List[Dict[str, Any]] = []
    batch_size = 32
    if not chunks:
        return index

    print(f"Embedding {len(chunks)} chunks in batches of {batch_size}...")
    for i in range(0, len(chunks), batch_size):
        print(f"Embedding chunks {i} to {min(i + batch_size, len(chunks))}...")
        batch = chunks[i : i + batch_size]
        embeddings = embeddings_model.embed_documents(batch)
        for chunk, vector in zip(batch, embeddings):
            index.append({"content": chunk, "embedding": np.asarray(vector)})
    return index


def cosine_similarity(a, b) -> float:
    """Cosine similarity, the dot product of the L2 normalized vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def retrieve(
    query: str,
    index: List[Dict[str, Any]],
    embeddings_model,
    top_k: int = 3,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """Score the query against every chunk and return the strongest matches."""
    if not index:
        return []

    query_vector = np.asarray(embeddings_model.embed_query(query))

    scored = [
        {
            "content": item["content"],
            "score": cosine_similarity(query_vector, item["embedding"]),
        }
        for item in index
    ]
    scored.sort(key=lambda row: row["score"], reverse=True)
    return [row for row in scored[:top_k] if row["score"] >= min_score]


### Augmentation
def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Render retrieved chunks as numbered, delimited blocks."""
    blocks = []
    for position, chunk in enumerate(chunks, start=1):
        score = chunk.get("score", 0.0)
        blocks.append(
            f"[{position}] (relevance {score:.3f})\n{chunk['content'].strip()}"
        )
    return "\n\n".join(blocks)


### Generation
def generate_answer(query: str, context_chunks: List[Dict[str, Any]], llm) -> str:
    """Call the generator with a grounded prompt, or abstain when evidence is empty."""
    if not context_chunks:
        return ABSTENTION_MESSAGE

    prompt = (
        f"{GROUNDED_SYSTEM_PROMPT}\n\n"
        f"CONTEXT\n{build_context(context_chunks)}\n\n"
        f"QUESTION\n{query}\n\n"
        f"ANSWER"
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


if __name__ == "__main__":
    from langchain_ollama import OllamaEmbeddings, ChatOllama

    embedder = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434" # Optional: default is localhost
    )

    llm = ChatOllama(
        model="llama3.1",
        temperature=0.2,
        base_url="http://localhost:11434"
    )

    # Indexing
    print("Loading documents...")
    file_paths = ['./arxiv_pdfs/pdfs/2411.05803v4.pdf', './arxiv_pdfs/pdfs/2412.20245v4.pdf']
    loaded_documents = []

    for file_path in file_paths:
        with open(file_path, 'rb') as file:
            loaded_documents.append(load_document(file))

    all_chunks = []
    for document in loaded_documents:
        all_chunks.extend(split_text(document))

    print(f"Total chunks created: {len(all_chunks)}")
    print("Building index...")
    index = build_simple_index(all_chunks, embedder)

    # Retrieval and Generation
    queries = [
        "Which stock has the highest percentage of trading days with both high liquidity diffusion and high liquidity jump?",
        "What is the purpose of manipulative traders in the hypothetical trading environment?",
        "Why might multiple predictors be beneficial in a regression model despite potential issues with multicollinearity?",
        "How do different predictors influence the dependent variable in a 4D Ridge Regression model for TM polarization?",
        # Irrelevant query to test abstention
        "What is the primary purpose of the Dvorak technique in tropical cyclone forecasting?",
        "What analogy is used to justify the scalability assumption in economics?"
    ]

    for query in queries:
        print(f"Processing query: {query}")
        retrieved_chunks = retrieve(query, index, embedder, top_k=3, min_score=0.0)
        answer = generate_answer(query, retrieved_chunks, llm)
        print(f"Query: {query}\nAnswer: {answer}\n{'-'*80}\n\nRetrieved Chunks:\n{retrieved_chunks}\n{'='*80}\n")