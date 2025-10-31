"""이전 유틸리티 호환을 위해 남겨둔 DB 헬퍼 클래스."""

import datetime
import json
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from config import settings


class DatabaseManager:
    def __init__(self):
        self.dsn = settings.database_url

    def _default_serializer(self, obj):
        if isinstance(obj, datetime.date):
            return obj.isoformat()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    def get_connection(self):
        return psycopg2.connect(self.dsn)

    def execute_query(self, query: str) -> str:
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query)
                    result_data = cur.fetchall()
            result_list = [dict(row) for row in result_data]
            return json.dumps(
                result_list,
                ensure_ascii=False,
                indent=2,
                default=self._default_serializer,
            )
        except Exception as exc:
            return f"DATABASE ERROR: {exc}"

    def test_connection(self) -> bool:
        try:
            with self.get_connection() as conn:
                return not conn.closed
        except Exception:
            return False

    def get_table_info(self) -> Optional[dict]:
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_name = 'game'
                        """
                    )
                    columns = cur.fetchall()
                    cur.execute("SELECT COUNT(*) as count FROM game")
                    count = cur.fetchone()
            return {"columns": columns, "record_count": count["count"]}
        except Exception:
            return None


db_manager = DatabaseManager()
