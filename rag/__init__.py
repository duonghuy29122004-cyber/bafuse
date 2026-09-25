"""BaFuse v2 RAG explanation layer."""
from .knowledge_base import DEGRADATION_KNOWLEDGE_BASE, get_entries_for_mode
from .retriever import DegradationExplainer

__all__ = ["DEGRADATION_KNOWLEDGE_BASE", "get_entries_for_mode", "DegradationExplainer"]
