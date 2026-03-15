"""
Shared rate limiter for the Instacart recommendation API.

Used by main app and route modules for consistent rate limiting.
"""

from __future__ import annotations

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

_default_rate_limit = os.getenv("RATE_LIMIT", "100/minute")
limiter = Limiter(key_func=get_remote_address, default_limits=[_default_rate_limit])
