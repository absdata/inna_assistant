-- File chunks table for large files
create table if not exists inna_file_chunks (
    id bigint primary key generated always as identity,
    message_id bigint references inna_messages(id) on delete cascade,
    chunk_index int not null,
    chunk_content text not null,
    source_type text not null default 'file',
    title text,
    section_title text,
    total_chunks int not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    unique(message_id, chunk_index)
); 