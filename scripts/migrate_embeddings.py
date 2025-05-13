#!/usr/bin/env python3
"""
Migration script to handle the transition from 1536D to 2000D embeddings.
This script will:
1. Update the database schema
2. Migrate existing embeddings
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from services.database import db_service
from utils.embedding_compressor import compressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def alter_table_columns():
    """Alter table columns to use 2000D vectors."""
    try:
        logger.info("Altering table columns to use 2000D vectors...")
        
        # Drop existing functions first to avoid dependency issues
        drop_functions_sql = """
        DROP FUNCTION IF EXISTS match_messages(vector, float, int);
        DROP FUNCTION IF EXISTS match_agent_memories(vector, text, float, int);
        DROP FUNCTION IF EXISTS match_agent_memories(vector, text, float, int, timestamp with time zone, timestamp with time zone);
        """
        db_service.client.query(drop_functions_sql).execute()
        logger.info("Dropped existing functions")
        
        # Alter tables to use 2000D vectors
        alter_tables_sql = """
        -- First, drop existing indexes that depend on the vector columns
        DROP INDEX IF EXISTS inna_message_embeddings_embedding_idx;
        
        -- Alter the tables to use 2000D vectors
        ALTER TABLE inna_message_embeddings 
        ALTER COLUMN embedding TYPE vector(2000) USING embedding::vector(2000);
        
        ALTER TABLE inna_tasks 
        ALTER COLUMN embedding TYPE vector(2000) USING 
        CASE 
            WHEN embedding IS NULL THEN NULL 
            ELSE embedding::vector(2000)
        END;
        
        ALTER TABLE inna_summaries 
        ALTER COLUMN embedding TYPE vector(2000) USING 
        CASE 
            WHEN embedding IS NULL THEN NULL 
            ELSE embedding::vector(2000)
        END;
        
        ALTER TABLE inna_agent_memory 
        ALTER COLUMN embedding TYPE vector(2000) USING embedding::vector(2000);
        
        -- Recreate the index
        CREATE INDEX inna_message_embeddings_embedding_idx 
        ON inna_message_embeddings 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """
        
        db_service.client.query(alter_tables_sql).execute()
        logger.info("Successfully altered table columns")
        
    except Exception as e:
        logger.error(f"Error altering table columns: {str(e)}", exc_info=True)
        raise

async def migrate_embeddings_in_table(table_name: str, embedding_column: str = "embedding"):
    """Migrate embeddings in a specific table."""
    try:
        logger.info(f"Migrating embeddings in {table_name}...")
        
        # Get all records with embeddings
        result = db_service.client.table(table_name)\
            .select("id", embedding_column)\
            .execute()
        
        if not result.data:
            logger.info(f"No records found in {table_name}")
            return
        
        migrated_count = 0
        skipped_count = 0
        error_count = 0
        
        for record in result.data:
            try:
                embedding = record[embedding_column]
                if embedding is None:
                    skipped_count += 1
                    continue
                
                # Check if embedding needs migration
                if len(embedding) != 2000:
                    # Compress the embedding to 2000D
                    compressed = compressor.compress(embedding)
                    
                    # Update the record
                    db_service.client.table(table_name)\
                        .update({embedding_column: compressed})\
                        .eq("id", record["id"])\
                        .execute()
                    
                    migrated_count += 1
                else:
                    skipped_count += 1
                
            except Exception as e:
                logger.error(f"Error migrating record {record['id']}: {str(e)}")
                error_count += 1
        
        logger.info(
            f"Migration results for {table_name}:\n"
            f"- Migrated: {migrated_count}\n"
            f"- Skipped: {skipped_count}\n"
            f"- Errors: {error_count}"
        )
        
    except Exception as e:
        logger.error(f"Error migrating {table_name}: {str(e)}", exc_info=True)
        raise

async def migrate_database():
    """Run the database migration."""
    try:
        logger.info("Starting database migration...")
        
        # First, alter table columns
        await alter_table_columns()
        
        # Update schema to use 2000D vectors
        schema_path = project_root / "database" / "schema.sql"
        with open(schema_path, "r") as f:
            schema_sql = f.read()
        
        # Execute schema updates
        logger.info("Updating database schema...")
        db_service.client.query(schema_sql).execute()
        logger.info("Schema updated successfully")
        
        # Migrate embeddings in each table
        tables = [
            "inna_message_embeddings",
            "inna_tasks",
            "inna_summaries",
            "inna_agent_memory"
        ]
        
        for table in tables:
            await migrate_embeddings_in_table(table)
        
        logger.info("Migration completed successfully")
        
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(migrate_database())
    except KeyboardInterrupt:
        logger.info("Migration interrupted by user")
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        sys.exit(1) 