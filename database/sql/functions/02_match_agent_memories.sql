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
    metadata jsonb,
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
        inna_agent_memory.metadata,
        1 - (inna_agent_memory.embedding <=> query_embedding) as similarity
    from inna_agent_memory
    where 
        (agent_role is null or inna_agent_memory.agent_role = agent_role)
        and 1 - (inna_agent_memory.embedding <=> query_embedding) > match_threshold
        and (start_time is null or inna_agent_memory.created_at >= start_time)
        and (end_time is null or inna_agent_memory.created_at <= end_time)
    order by inna_agent_memory.embedding <=> query_embedding
    limit match_count;
end;
$$; 