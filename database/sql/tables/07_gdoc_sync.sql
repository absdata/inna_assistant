-- Google Doc sync status table
create table if not exists inna_gdoc_sync (
    id bigint primary key generated always as identity,
    doc_id text not null,
    doc_type text not null, -- 'summary', 'task', etc.
    reference_id bigint not null, -- ID from respective table
    last_synced_at timestamp with time zone,
    sync_status text not null default 'pending',
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
); 