-- Summaries table for conversation summaries
create table if not exists inna_summaries (
    id bigint primary key generated always as identity,
    chat_id bigint not null,
    summary text not null,
    embedding vector(2000),
    metadata jsonb default '{}'::jsonb,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
); 