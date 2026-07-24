from .base import MindMemOSError


class ApiError(MindMemOSError):
    """Base class for HTTP API layer errors.

    Carries an HTTP status code so the FastAPI exception handler can render a
    consistent error envelope without per-endpoint try/except.
    """

    status_code: int = 500
    code: str = "internal_error"

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details=None,
    ):
        super().__init__(message or self.__class__.__name__)
        self.message = message or self.__class__.__name__
        self.details = details
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code


class AuthenticationError(ApiError):
    """Raised when the api_key is missing, malformed, or not recognized."""

    status_code = 401
    code = "unauthenticated"


class BadRequestError(ApiError):
    """Raised when a request is syntactically valid but missing required business fields."""

    status_code = 400
    code = "bad_request"


class InvalidFilterError(ApiError):
    """Raised when a search filter DSL payload is malformed or uses a disallowed field."""

    status_code = 400
    code = "invalid_filter"


class PermissionDeniedError(ApiError):
    """Raised when an authenticated caller lacks the required permission."""

    status_code = 403
    code = "permission_denied"


class BadRequestError(ApiError):
    """Raised when a request body is well-formed but semantically invalid."""

    status_code = 400
    code = "bad_request"


class ResourceNotFoundError(ApiError):
    """Raised when a referenced resource does not exist in the caller's project."""

    status_code = 404
    code = "not_found"
