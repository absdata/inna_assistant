-- Function to match messages based on embedding similarity
CREATE OR REPLACE FUNCTION match_messages(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.3,
    match_count int DEFAULT 10,
    section_filter text DEFAULT NULL
)
RETURNS TABLE (
    id bigint,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT DISTINCT ON (e.message_id)
        e.message_id as id,
        1 - (e.embedding <=> query_embedding) as similarity
    FROM inna_message_embeddings e
    WHERE 1 - (e.embedding <=> query_embedding) > match_threshold
    ORDER BY e.message_id, similarity DESC
    LIMIT match_count;
END;
$$; 