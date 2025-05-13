-- Enable required extensions
create extension if not exists vector;

-- Include table definitions
\i tables/01_messages.sql
\i tables/02_file_chunks.sql
\i tables/03_message_embeddings.sql
\i tables/04_tasks.sql
\i tables/05_summaries.sql
\i tables/06_agent_memory.sql
\i tables/07_gdoc_sync.sql

-- Include indexes
\i indexes/01_message_indexes.sql
\i indexes/02_task_indexes.sql
\i indexes/03_memory_indexes.sql

-- Include functions
\i functions/01_match_messages.sql
\i functions/02_match_agent_memories.sql 