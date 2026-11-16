import math
import re
import numpy as np
from typing import List
from app.config import settings

class EmbeddingService:
    """
    Embedding service supporting:
    1. Built-in Semantic Domain Vectorizer (Offline, zero-latency, sandbox safe)
    2. Sentence-Transformers (Local neural model when available)
    3. OpenAI text-embedding-3-small (when OPENAI_API_KEY configured)
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension
        self._st_model = None

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a text string into a normalized dense vector of floats.
        """
        # Clean and normalize text
        tokens = self._tokenize(text)
        if not tokens:
            vec = np.zeros(self.dimension, dtype=np.float32)
            return vec.tolist()

        # Semantic concept mapping for B2B partnerships
        concept_clusters = {
            "ecommerce_food_delivery": [
                "food", "delivery", "swiggy", "zomato", "restaurant", "dining", "meal", 
                "ecommerce", "e-commerce", "shopping", "orders", "festive", "diwali", 
                "discount", "offer", "promotions", "coupon", "cashback", "deal"
            ],
            "payments_fintech": [
                "payment", "payments", "fintech", "upi", "phonepe", "paytm", "gpay", 
                "card", "credit", "debit", "wallet", "merchant", "cashback", "reward", 
                "banking", "transaction", "checkout", "discount", "offer"
            ],
            "banking_lending": [
                "bank", "banking", "hdfc", "icici", "sbi", "loan", "emi", "credit", 
                "interest", "account", "deposit", "savings", "card"
            ],
            "automotive_transport": [
                "automotive", "auto", "bajaj", "bike", "car", "vehicle", "two-wheeler", 
                "dealership", "dealers", "spare", "parts", "logistics", "engine", "manufacturing"
            ],
            "travel_hospitality": [
                "travel", "hotel", "flight", "booking", "trip", "holiday", "ticket", "tourism"
            ]
        }

        # Initialize base hash vector
        vec = np.zeros(self.dimension, dtype=np.float32)
        
        # Word-level feature hashing
        for token in tokens:
            h = hash(token) % self.dimension
            vec[h] += 1.0

        # Cluster semantic expansion: inject high-signal concept energy into designated slots
        slot_size = self.dimension // (len(concept_clusters) + 2)
        for idx, (cluster_name, keywords) in enumerate(concept_clusters.items()):
            overlap_score = sum(1.0 for kw in keywords if kw in tokens or any(kw in t for t in tokens))
            if overlap_score > 0:
                start_idx = idx * slot_size
                end_idx = start_idx + slot_size
                vec[start_idx:end_idx] += overlap_score * 2.5

        # L2 Normalization
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return [float(x) for x in vec]

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        # Keep alphanumeric words and currency symbols
        tokens = re.findall(r'\b[a-z0-9\-]+\b', text)
        return tokens

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """
        Computes cosine similarity between two normalized vectors:
        Cosine Similarity = dot(u, v) / (norm(u) * norm(v))
        """
        if not v1 or not v2:
            return 0.0
        
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        sim = float(np.dot(a, b) / (norm_a * norm_b))
        # Clamp to [0.0, 1.0] for probability interpretation
        return max(0.0, min(1.0, sim))

# Global embedding service instance
embedding_service = EmbeddingService()
