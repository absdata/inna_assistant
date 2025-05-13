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