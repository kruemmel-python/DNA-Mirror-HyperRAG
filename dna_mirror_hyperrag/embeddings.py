"""Deterministic text embedding utilities for lightweight integrations."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .core import _tokenize


@dataclass
class TextEmbedder:
    """Generate deterministic embeddings via hashed token projections.

    The implementation intentionally avoids heavyweight ML dependencies while
    still providing stable float vectors that downstream systems such as LM
    Studio can ingest via an OpenAI-compatible surface. Tokens are projected
    into a fixed-size vector space using a signed hashing trick and
    L2-normalised to unit length.
    """

    dimension: int = 384
    seed: int = 13

    def _hash_bytes(self, token: str) -> bytes:
        data = f"{self.seed}:{token}".encode("utf-8", "ignore")
        return hashlib.sha256(data).digest()

    def _accumulate(self, tokens: Sequence[str]) -> list[float]:
        if not tokens:
            return [0.0] * self.dimension

        vector = [0.0] * self.dimension
        for tok in tokens:
            digest = self._hash_bytes(tok)
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            magnitude = (int.from_bytes(digest[5:7], "big") / 65535.0) + 0.5
            vector[bucket] += sign * magnitude

        norm = math.sqrt(sum(v * v for v in vector))
        if norm:
            return [v / norm for v in vector]
        return vector

    def embed_tokens(self, tokens: Sequence[str]) -> list[float]:
        """Return an L2-normalised embedding for the supplied tokens."""

        return self._accumulate(tokens)

    def embed_text(self, text: str | Iterable[str]) -> list[float]:
        """Embed a string or iterable of strings into a single vector."""

        if isinstance(text, str):
            tokens = _tokenize(text)
        else:
            tokens: list[str] = []
            for chunk in text:
                tokens.extend(_tokenize(chunk))
        return self.embed_tokens(tokens)

