from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ============================================================
# SAMPLE TEXT
# ============================================================

text = """
Artificial Intelligence (AI) enables machines to perform tasks
that normally require human intelligence.

Machine Learning is a subset of AI. It allows systems to learn
patterns from data and make predictions.

Deep Learning is a subset of Machine Learning. It uses neural
networks with multiple layers to learn complex patterns.

Retrieval-Augmented Generation (RAG) combines information
retrieval with Large Language Models. It retrieves relevant
documents and uses them to generate grounded answers.

Vector databases store embeddings and support similarity search.
Examples include Qdrant, Chroma, and FAISS.
"""


def display_chunks(title, chunks):
    print(f"\n{'=' * 60}")
    print(f"{title} | Total chunks: {len(chunks)}")
    print("=" * 60)

    for i, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {i}:")
        print(chunk)
        print("-" * 40)


# ============================================================
# 1. FIXED-SIZE CHUNKING
# ============================================================
# Splits text into chunks of approximately 100 characters.
# May split sentences or ideas.

fixed = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=0,
)

fixed_chunks = fixed.split_text(text)
display_chunks("FIXED-SIZE CHUNKING", fixed_chunks)


# ============================================================
# 2. PARAGRAPH / SEPARATOR CHUNKING
# ============================================================
# Splits text using paragraph boundaries.
# Each paragraph is kept together when it fits the chunk size.

paragraph = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=200,
    chunk_overlap=0,
)

paragraph_chunks = paragraph.split_text(text)
display_chunks("PARAGRAPH CHUNKING", paragraph_chunks)


# ============================================================
# 3. RECURSIVE CHUNKING
# ============================================================
# Tries separators in order:
# paragraphs -> lines -> spaces -> characters.
# Preferred general-purpose option for RAG.

recursive = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    separators=["\n\n", "\n", " ", ""],
)

recursive_chunks = recursive.split_text(text)
display_chunks("RECURSIVE CHUNKING", recursive_chunks)
