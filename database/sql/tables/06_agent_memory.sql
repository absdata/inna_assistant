-- Agent memory table for long-term context
create table if not exists inna_agent_memory (
    id bigint primary key generated always as identity,
    chat_id bigint not null,
    agent_role text not null,
    context text not null,
    embedding vector(2000) not null,
    relevance_score float not null default 1.0,
    metadata jsonb default '{}'::jsonb,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
); 