-- Tasks table for planning and roadmap
create table if not exists inna_tasks (
    id bigint primary key generated always as identity,
    chat_id bigint not null,
    title text not null,
    description text,
    status text not null default 'pending',
    priority int not null default 1,
    due_date timestamp with time zone,
    embedding vector(2000),
    metadata jsonb default '{}'::jsonb,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
); 