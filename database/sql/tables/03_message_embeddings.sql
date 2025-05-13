-- Message embeddings table with vector search capability
create table if not exists inna_message_embeddings (
    id bigint primary key generated always as identity,
    message_id bigint references inna_messages(id) on delete cascade,
    chat_id bigint not null,
    text text not null,
    embedding vector(2000) not null,
    chunk_index int,  -- NULL for regular messages, index number for file chunks
    section_title text,  -- NULL for regular messages, section title for file chunks
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
); 