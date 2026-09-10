"""Persistent, time-aware stock knowledge for GMQ.

The knowledge layer is intentionally separate from the predictive models. It
stores identity, observed market history, future-known events and GMQ's own
outcomes so the feature engine can expose that context to the decision stack.
"""
from .store import KnowledgeStore, StockKnowledge, FutureEvent
from .pretrade import PreTradeKnowledge, PreTradeVerdict

__all__ = [
    "KnowledgeStore",
    "StockKnowledge",
    "FutureEvent",
    "PreTradeKnowledge",
    "PreTradeVerdict",
]
