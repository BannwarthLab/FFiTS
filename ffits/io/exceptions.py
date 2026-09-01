#!/bin/python
"""Custom exceptions used across the ffits package."""


class ConvergenceError(Exception):
    """Raised when an iterative calculation (e.g. optimization or fitting) fails to converge."""

    def __init__(self, message):
        """Store the error message."""
        self.message = message
        super().__init__(self.message)
