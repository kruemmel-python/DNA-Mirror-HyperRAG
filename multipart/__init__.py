"""Minimal stub of the :mod:`multipart` package for local testing.

The real project depends on :mod:`python-multipart` to process incoming file
uploads. Our automated test environment, however, does not ship the optional
dependency. FastAPI performs an import-time check for :mod:`multipart` when a
route uses file parameters, therefore we provide this tiny shim so the module
resolves during tests. Users running the Streamlit or FastAPI app in production
should install ``python-multipart`` to benefit from the full implementation.
"""

__version__ = "0.0.0-test"

