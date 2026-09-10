"""Compatibility wrapper for the Perplexity finance/news integration."""
from .perplexity_client import NewsItem, NewsStore, PerplexityClient, PerplexityNewsMemory

__all__ = ["NewsItem", "NewsStore", "PerplexityClient", "PerplexityNewsMemory"]
