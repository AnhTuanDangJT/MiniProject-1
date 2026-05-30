"""
semantic_search.py
------------------
Handles knowledge base loading and semantic search using sentence embeddings.

Dependencies:
    pip install pandas sentence-transformers scikit-learn
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSearch:
    """
    Loads a Q&A knowledge base from a CSV file and answers user queries
    by finding the most semantically similar question using cosine similarity
    on sentence embeddings.
    """

    def __init__(self, csv_file: str):
        """
        Initialize the SemanticSearch system.

        Args:
            csv_file (str): Path to the CSV file containing 'question' and 'answer' columns.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If required columns are missing from the CSV.
        """
        # --- Load the knowledge base ---
        try:
            self.data = pd.read_csv(csv_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Knowledge base file not found: {csv_file}")

        # Validate that required columns exist
        if "question" not in self.data.columns or "answer" not in self.data.columns:
            raise ValueError("CSV file must contain 'question' and 'answer' columns.")

        self.questions = self.data["question"].tolist()
        self.answers = self.data["answer"].tolist()

        # --- Load the pre-trained sentence embedding model ---
        # "all-MiniLM-L6-v2" is a lightweight model that maps sentences to
        # 384-dimensional vectors, balancing speed and accuracy well.
        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Pre-compute embeddings for all knowledge base questions.
        # We do this once at startup so each query is fast.
        print(f"Encoding {len(self.questions)} knowledge base questions...")
        self.question_embeddings = self.model.encode(self.questions)
        print("Semantic search ready.\n")

    def get_best_answer(self, user_question: str) -> tuple[str, float]:
        """
        Find the most relevant answer for a user's question.

        How it works:
            1. Convert the user's question into an embedding (a vector of numbers).
            2. Compute cosine similarity between the user embedding and all
               stored question embeddings.
            3. Return the answer whose question embedding is closest to the
               user's question embedding.

        Args:
            user_question (str): The question typed by the user.

        Returns:
            tuple[str, float]: The best matching answer and its similarity score (0.0 - 1.0).
        """
        # Step 1: Embed the user's question
        user_embedding = self.model.encode([user_question])

        # Step 2: Compute cosine similarity against all stored questions
        # cosine_similarity returns a 2D array; we take row 0
        similarities = cosine_similarity(user_embedding, self.question_embeddings)

        # Step 3: Pick the index with the highest similarity score
        best_index = similarities.argmax()
        best_score = float(similarities[0][best_index])

        return self.answers[best_index], best_score


# ---------------------------------------------------------------------------
# Quick standalone test — run this file directly to verify everything works.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    searcher = SemanticSearch("knowledgebase.csv")

    test_questions = [
        "I forgot my password, what do I do?",
        "How can I register for my classes?",
        "I need to pay my fees",
        "Where do I go to get my ID?",
    ]

    print("=== Semantic Search Test ===\n")
    for q in test_questions:
        answer, score = searcher.get_best_answer(q)
        print(f"Q: {q}")
        print(f"A: {answer}")
        print(f"   (similarity score: {score:.4f})\n")
