class SHGException(Exception):
    """Parent expection type for all custom exceptions in this module."""
    pass


class SHGValueError(SHGException):
    """ Invalid value for a parameter. """
    pass


class SHGIndexError(SHGException):
    """ Invalid index for an iterable. """
    pass


class SHGGeometryError(SHGException):
    """ Invalid or not expected geometry """
    pass


class SHGHierarchyError(SHGException):
    """ Invalid or not expected hierarchy """
    pass


class SHGPlannerError(SHGException):
    """ Invalid or no path found by planner """
    pass
