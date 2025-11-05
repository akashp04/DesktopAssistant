"""
Custom exceptions for Desktop Assistant
"""

class DesktopAssistantError(Exception):
    pass

class ServiceError(DesktopAssistantError):
    pass

class ValidationError(DesktopAssistantError):
    pass

class StorageError(DesktopAssistantError):
    pass