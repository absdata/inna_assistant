-- Function to match messages using vector similarity
create or replace function match_messages(
    query_embedding vector(2000),
    match_threshold float,
    match_count int,
    section_filter text default null
)
returns table (
    id bigint,
    similarity float,
    text text,
    chunk_index int,
    section_title text
)
language plpgsql
as $$
begin
    return query
    select
        e.message_id as id,
        1 - (e.embedding <=> query_embedding) as similarity,
        e.text,
        e.chunk_index,
        e.section_title
    from
        inna_message_embeddings e
    where
        case
            when section_filter is not null then
                e.section_title = section_filter
            else
                true
        end
        and 1 - (e.embedding <=> query_embedding) > match_threshold
    order by
        e.embedding <=> query_embedding
    limit match_count;
end;
$$; 