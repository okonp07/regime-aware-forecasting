"""Custom exceptions for the RAF application."""


class RAFError(Exception):
    """Base exception for RAF."""


class DataFetchError(RAFError):
    """Failed to fetch market data."""


class ValidationError(RAFError):
    """Data validation failed."""


class ModelFitError(RAFError):
    """Model fitting failed."""


class RunNotFoundError(RAFError):
    """Requested run does not exist."""


class ExportError(RAFError):
    """Export generation failed."""
