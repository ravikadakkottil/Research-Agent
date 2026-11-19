"""
Shared pytest setup.

Tests never call real APIs. A dummy GROQ_API_KEY env var means Settings()
can be created even without a .env (e.g. in CI). Real env vars beat .env
values in pydantic-settings, so your real key is never used by tests.
"""

import os

os.environ["GROQ_API_KEY"] = "test-key-not-used"
os.environ.pop("OPENALEX_API_KEY", None)
# Never touch the real on-disk caches from tests.
os.environ["CACHE_ENABLED"] = "false"
os.environ["EMBEDDING_CACHE_ENABLED"] = "false"
# No real Gemini calls: rerank tests inject a fake embedder.
os.environ["GEMINI_API_KEY"] = ""
