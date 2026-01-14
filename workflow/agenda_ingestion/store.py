
import logging
import os
from typing import List
from tools.oracle_client import OracleClient

logger = logging.getLogger("oracle_storage")



def store_agenda_items(meeting_id: str, items: List[str]):
    """
    Stores the list of agenda items into the Oracle Database.
    Table: MEETING_AGENDA_ITEMS (MEETING_ID, ITEM_TEXT)
    """
    if not items:
        logger.info(f"[{meeting_id}] No items to store.")
        return

    client = OracleClient()
    try:
        # 1️⃣ Delete existing agenda items
        delete_sql = """
            DELETE FROM MEETING_AGENDA_ITEMS
            WHERE MEETING_ID = :meeting_id
        """
        client.execute_write(delete_sql, {"meeting_id": meeting_id})
        logger.info(f"[{meeting_id}] Cleared existing agenda items (if any).")

        # 2️⃣ Insert new agenda items
        insert_sql = """
            INSERT INTO MEETING_AGENDA_ITEMS (MEETING_ID, ITEM_TEXT)
            VALUES (:meeting_id, :item_text)
        """

        data = [
            {
                "meeting_id": meeting_id,
                "item_text": item
            }
            for item in items
        ]

        client.execute_many(insert_sql, data)

        logger.info(
            f"[{meeting_id}] Successfully stored {len(items)} agenda items in Oracle DB."
        )

    except Exception as e:
        logger.exception(f"[{meeting_id}] Store Error")


def update_agenda_status(meeting_id: str, item_text: str, status: str, summary: str):
    """
    Updates the status and summary for a specific agenda item in the Oracle Database.
    Matches based on MEETING_ID and ITEM_TEXT (exact match for now).
    """
    client = OracleClient()
    try:
        sql = """
            UPDATE MEETING_AGENDA_ITEMS 
            SET STATUS = :1, SUMMARY = :2 
            WHERE MEETING_ID = :3 AND DBMS_LOB.COMPARE(ITEM_TEXT, :4) = 0
        """
        # Note: ITEM_TEXT LOB comparison might vary.
        
        rowcount = client.execute_write(sql, [status, summary, meeting_id, item_text])
        
        if rowcount > 0:
            logger.info(f"[{meeting_id}] Updated status for item: {item_text[:30]}...")
        else:
            logger.warning(f"[{meeting_id}] No matching item found to update: {item_text[:30]}...")

    except Exception as e:
        logger.error(f"[{meeting_id}] Update Error: {e}")

def fetch_agenda_items(meeting_id: str) -> List[str]:
    """
    Retrieves the list of agenda items for a specific meeting from the Oracle Database.
    """
    client = OracleClient()
    items = []

    try:
        sql = "SELECT ITEM_TEXT FROM MEETING_AGENDA_ITEMS WHERE MEETING_ID = :1 ORDER BY ID"
        rows = client.execute_query(sql, [meeting_id])
        
        for row in rows:
            item_text = row[0]
            if hasattr(item_text, 'read'):
                items.append(item_text.read())
            else:
                items.append(str(item_text))
                
        logger.info(f"[{meeting_id}] Fetched {len(items)} agenda items from DB.")
        return items

    except Exception as e:
        logger.error(f"[{meeting_id}] Fetch Error: {e}")
        return []