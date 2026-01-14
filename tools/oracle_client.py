import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("oracle_client")

try:
    import oracledb
    ORACLE_AVAILABLE = True
    oracledb.init_oracle_client()

except ImportError:
    ORACLE_AVAILABLE = False
    logger.warning("oracledb module not found. OracleClient will fail on connection attempts.")

class OracleClient:
    """
    Generic wrapper for Oracle Database interactions.
    """
    def __init__(self):
        self.user = os.environ.get("ORACLE_USER","CP_DWH")
        self.password = os.environ.get("ORACLE_PASSWORD","May#2022")
        self.dsn = os.environ.get("ORACLE_DSN","172.17.205.98:1541/nspcpdb1")
    def _get_connection(self):
        """
        Establishes and returns a new connection.
        """
        if not ORACLE_AVAILABLE:
            raise ImportError("oracledb library is not installed.")
        
        if not all([self.user, self.password, self.dsn]):
            raise ValueError("Missing Oracle DB environment variables (ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN).")
            
        conn = oracledb.connect(user=self.user, password=self.password, dsn=self.dsn)
        conn.outputtypehandler = self._output_type_handler
        return conn

    @staticmethod
    def _output_type_handler(cursor, name, default_type, size, precision, scale):
        """
        Automatically convert CLOBs to Strings to avoid LOB locator issues after connection close.
        """
        if default_type == oracledb.CLOB or default_type == oracledb.NCLOB:
            return cursor.var(str, arraysize=cursor.arraysize)

    def execute_query(self, sql: str, params: List[Any] = None) -> List[Any]:
        """
        Executes a SELECT query and returns all rows.
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(sql, params or [])
            return cursor.fetchall()
            
        except Exception as e:
            logger.error(f"Oracle Query Error: {e}")
            raise
        finally:
            if conn: conn.close()

    def execute_query_dicts(self, sql: str, params: List[Any] = None) -> List[Dict[str, Any]]:
        """
        Executes a SELECT query and returns rows as dictionaries.
        Keys are column names in lowercase.
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(sql, params or [])
            
            # Get columns
            if not cursor.description:
                return []
                
            columns = [col[0].lower() for col in cursor.description]
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                result.append(dict(zip(columns, row)))
                
            return result
            
        except Exception as e:
            logger.error(f"Oracle Query Dict Error: {e}")
            raise
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def execute_write(self, sql: str, params: List[Any] = None) -> int:
        """
        Executes a write operation (INSERT, UPDATE, DELETE) and commits.
        Returns the number of affected rows (if available/supported by driver, else 0).
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(sql, params or [])
            conn.commit()
            return cursor.rowcount
            
        except Exception as e:
            if conn: conn.rollback()
            logger.error(f"Oracle Write Error: {e}")
            raise
        finally:
            if cursor: cursor.close()
            if conn: conn.close()
            
    def execute_many(self, sql: str, data: List[Any]) -> None:
        """
        Executes a batch INSERT operation and commits.
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.executemany(sql, data)
            conn.commit()
            
        except Exception as e:
            if conn: conn.rollback()
            logger.error(f"Oracle Batch Write Error: {e}")
            raise
        finally:
            if cursor: cursor.close()
            
    def execute_query_formatted(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Executes a SELECT query, returns rows as dictionaries with lowercase keys,
        and converts datetime objects to ISO format strings.
        """
        rows = self.execute_query_dicts(sql, params)
        for record in rows:
            for k, v in record.items():
                if hasattr(v, "isoformat"):
                    record[k] = v.isoformat()
        return rows
