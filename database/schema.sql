-- SQL schema for Supabase tables

-- Enable the pgvector extension
create extension if not exists vector;

-- Messages table to store all Telegram messages
create table if not exists inna_messages (
    id bigint primary key generated always as identity,
    chat_id bigint not null,
    message_id bigint not null,
    user_id bigint not null,
    username text,
    text text,
    file_url text,
    file_content text,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    unique(chat_id, message_id)
);

-- File chunks table for large files
create table if not exists inna_file_chunks (
    id bigint primary key generated always as identity,
    message_id bigint references inna_messages(id) on delete cascade,
    chunk_index int not null,
    chunk_content text not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    unique(message_id, chunk_index)
);

-- Create index for faster chunk retrieval
create index if not exists inna_file_chunks_message_id_idx on inna_file_chunks(message_id);

-- Message embeddings table with vector search capability
create table if not exists inna_message_embeddings (
    id bigint primary key generated always as identity,
    message_id bigint references inna_messages(id) on delete cascade,
    chat_id bigint not null,
    text text not null,
    embedding vector(2000) not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Tasks table for planning and roadmap
create table if not exists inna_tasks (
    id bigint primary key generated always as identity,
    title text not null,
    description text,
    status text not null default 'pending',
    priority int not null default 1,
    due_date timestamp with time zone,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
    chat_id bigint not null,
    assigned_to text,
    parent_task_id bigint references inna_tasks(id) on delete set null
);

-- Weekly summaries table
create table if not exists inna_summaries (
    id bigint primary key generated always as identity,
    chat_id bigint not null,
    summary_type text not null, -- 'weekly', 'monthly', etc.
    content text not null,
    period_start timestamp with time zone not null,
    period_end timestamp with time zone not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    gdoc_url text -- URL to synced Google Doc
);

-- Agent memory for multi-agent system
create table if not exists inna_agent_memory (
    id bigint primary key generated always as identity,
    agent_role text not null, -- 'planner', 'doer', 'critic'
    chat_id bigint not null,
    context text not null,
    embedding vector(2000) not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    relevance_score float not null default 1.0
);

-- Google Docs sync status
create table if not exists inna_gdoc_sync (
    id bigint primary key generated always as identity,
    doc_id text not null,
    doc_type text not null, -- 'summary', 'task', etc.
    reference_id bigint not null, -- ID from respective table
    last_synced_at timestamp with time zone,
    sync_status text not null default 'pending',
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create indexes for faster vector similarity search
create index if not exists inna_message_embeddings_embedding_idx 
on inna_message_embeddings 
using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

create index if not exists inna_agent_memory_embedding_idx 
on inna_agent_memory 
using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

-- Create indexes for faster lookups
create index if not exists inna_messages_chat_id_idx on inna_messages(chat_id);
create index if not exists inna_message_embeddings_chat_id_idx on inna_message_embeddings(chat_id);
create index if not exists inna_tasks_chat_id_idx on inna_tasks(chat_id);
create index if not exists inna_summaries_chat_id_idx on inna_summaries(chat_id);
create index if not exists inna_agent_memory_chat_id_idx on inna_agent_memory(chat_id);

-- Function to match similar messages based on embedding
create or replace function match_messages(
    query_embedding vector(2000),
    match_threshold float,
    match_count int
)
returns table (
    id bigint,
    chat_id bigint,
    text text,
    similarity float
)
language sql stable
as $$
    select
        inna_message_embeddings.id,
        inna_message_embeddings.chat_id,
        inna_message_embeddings.text,
        1 - (inna_message_embeddings.embedding <=> query_embedding) as similarity
    from inna_message_embeddings
    where 1 - (inna_message_embeddings.embedding <=> query_embedding) > match_threshold
    order by inna_message_embeddings.embedding <=> query_embedding
    limit match_count;
$$;

-- Function to match similar agent memories
create or replace function match_agent_memories(
    query_embedding vector(2000),
    agent_role text,
    match_threshold float,
    match_count int
)
returns table (
    id bigint,
    chat_id bigint,
    context text,
    similarity float
)
language sql stable
as $$
    select
        inna_agent_memory.id,
        inna_agent_memory.chat_id,
        inna_agent_memory.context,
        1 - (inna_agent_memory.embedding <=> query_embedding) as similarity
    from inna_agent_memory
    where 
        inna_agent_memory.agent_role = agent_role
        and 1 - (inna_agent_memory.embedding <=> query_embedding) > match_threshold
    order by inna_agent_memory.embedding <=> query_embedding
    limit match_count;
$$;