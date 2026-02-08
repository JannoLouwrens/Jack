"""
Jack Server Routes.

Additional route modules for extended functionality.
"""

from jack.server.routes.database import router as database_router

__all__ = ["database_router"]
