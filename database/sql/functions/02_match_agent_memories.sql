-- Function to match agent memories with time filtering
create or replace function match_agent_memories(
    query_embedding vector(2000),
    agent_role text,
    match_threshold float,
    match_count int,
    start_time timestamp with time zone default null,
    end_time timestamp with time zone default null
)
returns table (
    id bigint,
    chat_id bigint,
    context text,
    memory_role text,
    embedding vector(2000),
    metadata jsonb,
    relevance_score float,
    created_at timestamp with time zone,
    similarity float
)
language plpgsql
as $$
begin
    return query
    select
        inna_agent_memory.id,
        inna_agent_memory.chat_id,
        inna_agent_memory.context,
        inna_agent_memory.agent_role as memory_role,
        inna_agent_memory.embedding,
        inna_agent_memory.metadata,
        inna_agent_memory.relevance_score,
        inna_agent_memory.created_at,
        1 - (inna_agent_memory.embedding <=> query_embedding) as similarity
    from inna_agent_memory
    where 
        ($2 is null or inna_agent_memory.agent_role = $2)
        and 1 - (inna_agent_memory.embedding <=> query_embedding) > match_threshold
        and (start_time is null or inna_agent_memory.created_at >= start_time)
        and (end_time is null or inna_agent_memory.created_at <= end_time)
    order by inna_agent_memory.embedding <=> query_embedding
    limit match_count;
end;
$$; 