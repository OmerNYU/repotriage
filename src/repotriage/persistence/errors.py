"""Persistence domain exceptions."""


class PersistenceError(RuntimeError):
    """Base class for persistence domain errors."""


class FeedbackValidationError(PersistenceError):
    """Raised when feedback request content fails domain validation."""


class DatabaseUnavailableError(PersistenceError):
    """Raised when the database cannot be reached at startup."""
