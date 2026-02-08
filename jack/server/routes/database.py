"""
Database Routes - MSSQL Integration.

Provides secure database access for the Jack agent.
Supports parameterized queries with sandboxed execution.
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from contextlib import asynccontextmanager

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/db", tags=["Database"])


# =============================================================================
# Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    """SQL query request."""
    query: str = Field(..., max_length=10000)
    params: Optional[Dict[str, Any]] = None
    max_rows: int = Field(default=1000, le=10000)
    timeout: float = Field(default=30.0, le=300.0)


class QueryResponse(BaseModel):
    """SQL query response."""
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    truncated: bool = False


class TableInfo(BaseModel):
    """Table metadata."""
    name: str
    schema_name: str
    columns: List[Dict[str, str]]
    row_count: Optional[int] = None


# =============================================================================
# Database Connection Manager
# =============================================================================

@dataclass
class DatabaseConnection:
    """MSSQL database connection wrapper."""

    connection_string: str
    _connection: Any = None

    async def connect(self) -> None:
        """Establish database connection."""
        try:
            import aioodbc

            self._connection = await aioodbc.connect(dsn=self.connection_string)
            logger.info("Database connected")
        except ImportError:
            # Fallback to sync pyodbc in thread
            import pyodbc
            import asyncio

            loop = asyncio.get_event_loop()
            self._connection = await loop.run_in_executor(
                None,
                lambda: pyodbc.connect(self.connection_string)
            )
            logger.info("Database connected (sync mode)")

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._connection:
            try:
                await self._connection.close()
            except:
                self._connection.close()
            self._connection = None
            logger.info("Database disconnected")

    async def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        max_rows: int = 1000,
    ) -> tuple:
        """
        Execute a SQL query.

        Args:
            query: SQL query with named parameters
            params: Parameter values
            max_rows: Maximum rows to return

        Returns:
            Tuple of (columns, rows)
        """
        if not self._connection:
            raise RuntimeError("Not connected to database")

        # Convert named params to positional for pyodbc
        if params:
            for key, value in params.items():
                query = query.replace(f":{key}", "?")
            param_values = list(params.values())
        else:
            param_values = []

        try:
            cursor = await self._connection.cursor()
            await cursor.execute(query, param_values)

            # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Fetch rows with limit
            rows = []
            while len(rows) < max_rows:
                row = await cursor.fetchone()
                if row is None:
                    break
                rows.append(list(row))

            # Check if truncated
            truncated = await cursor.fetchone() is not None

            await cursor.close()
            return columns, rows, truncated

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    async def get_tables(self) -> List[TableInfo]:
        """Get list of tables in database."""
        query = """
        SELECT
            t.TABLE_SCHEMA,
            t.TABLE_NAME,
            c.COLUMN_NAME,
            c.DATA_TYPE
        FROM INFORMATION_SCHEMA.TABLES t
        JOIN INFORMATION_SCHEMA.COLUMNS c
            ON t.TABLE_NAME = c.TABLE_NAME
            AND t.TABLE_SCHEMA = c.TABLE_SCHEMA
        WHERE t.TABLE_TYPE = 'BASE TABLE'
        ORDER BY t.TABLE_SCHEMA, t.TABLE_NAME, c.ORDINAL_POSITION
        """

        columns, rows, _ = await self.execute(query)

        tables: Dict[str, TableInfo] = {}
        for row in rows:
            schema, table, col_name, col_type = row
            key = f"{schema}.{table}"

            if key not in tables:
                tables[key] = TableInfo(
                    name=table,
                    schema_name=schema,
                    columns=[],
                )

            tables[key].columns.append({
                "name": col_name,
                "type": col_type,
            })

        return list(tables.values())


# Global connection pool (simple single connection for now)
_db_connection: Optional[DatabaseConnection] = None


async def get_db() -> DatabaseConnection:
    """Get database connection dependency."""
    global _db_connection

    if _db_connection is None:
        import os
        from jack.server.config import DatabaseConfig

        config = DatabaseConfig.from_env()
        _db_connection = DatabaseConnection(config.connection_string)
        await _db_connection.connect()

    return _db_connection


# =============================================================================
# Query Validation
# =============================================================================

DANGEROUS_KEYWORDS = [
    "DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "EXEC",
    "EXECUTE", "INSERT", "UPDATE", "MERGE", "GRANT", "REVOKE",
    "xp_", "sp_", "--", ";--", "/*", "*/"
]


def validate_query(query: str, allow_writes: bool = False) -> None:
    """
    Validate SQL query for safety.

    Args:
        query: SQL query string
        allow_writes: Whether to allow write operations

    Raises:
        HTTPException: If query is dangerous
    """
    query_upper = query.upper()

    if not allow_writes:
        for keyword in DANGEROUS_KEYWORDS:
            if keyword.upper() in query_upper:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dangerous SQL keyword detected: {keyword}",
                )

    # Check for multiple statements
    if query.count(";") > 1:
        raise HTTPException(
            status_code=400,
            detail="Multiple SQL statements not allowed",
        )


# =============================================================================
# Routes
# =============================================================================

@router.post("/query", response_model=QueryResponse)
async def execute_query(
    request: QueryRequest,
    db: DatabaseConnection = Depends(get_db),
) -> QueryResponse:
    """
    Execute a read-only SQL query.

    Parameters are passed securely to prevent SQL injection.
    """
    validate_query(request.query, allow_writes=False)

    try:
        columns, rows, truncated = await db.execute(
            request.query,
            request.params,
            request.max_rows,
        )

        return QueryResponse(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
        )

    except Exception as e:
        raise HTTPException(500, f"Query failed: {str(e)}")


@router.get("/tables", response_model=List[TableInfo])
async def list_tables(
    db: DatabaseConnection = Depends(get_db),
) -> List[TableInfo]:
    """Get list of all tables in database."""
    try:
        return await db.get_tables()
    except Exception as e:
        raise HTTPException(500, f"Failed to list tables: {str(e)}")


@router.get("/tables/{schema}/{table}")
async def describe_table(
    schema: str,
    table: str,
    db: DatabaseConnection = Depends(get_db),
) -> Dict[str, Any]:
    """Get detailed information about a table."""
    query = """
    SELECT
        c.COLUMN_NAME,
        c.DATA_TYPE,
        c.CHARACTER_MAXIMUM_LENGTH,
        c.IS_NULLABLE,
        c.COLUMN_DEFAULT
    FROM INFORMATION_SCHEMA.COLUMNS c
    WHERE c.TABLE_SCHEMA = :schema AND c.TABLE_NAME = :table
    ORDER BY c.ORDINAL_POSITION
    """

    try:
        columns, rows, _ = await db.execute(
            query,
            {"schema": schema, "table": table},
        )

        if not rows:
            raise HTTPException(404, "Table not found")

        return {
            "schema": schema,
            "table": table,
            "columns": [
                {
                    "name": row[0],
                    "type": row[1],
                    "max_length": row[2],
                    "nullable": row[3] == "YES",
                    "default": row[4],
                }
                for row in rows
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to describe table: {str(e)}")


@router.get("/health")
async def database_health() -> Dict[str, Any]:
    """Check database connectivity."""
    try:
        db = await get_db()
        columns, rows, _ = await db.execute("SELECT 1 as health")
        return {"status": "connected", "healthy": True}
    except Exception as e:
        return {"status": "error", "healthy": False, "error": str(e)}
