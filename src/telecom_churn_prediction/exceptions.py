class DatasetValidationError(Exception):
    """Raised when the dataset does not match expected schema."""


class ArtifactNotFoundError(Exception):
    """Raised when required model artifacts are missing."""


class PredictionError(Exception):
    """Raised when prediction fails."""