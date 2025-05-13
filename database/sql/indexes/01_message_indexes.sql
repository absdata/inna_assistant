-- Create indexes for message-related tables
create index if not exists inna_messages_chat_id_idx on inna_messages(chat_id);

-- File chunks indexes
create index if not exists inna_file_chunks_message_id_idx on inna_file_chunks(message_id);
create index if not exists inna_file_chunks_section_title_idx on inna_file_chunks(section_title);

-- Message embeddings indexes
create index if not exists inna_message_embeddings_chat_id_idx on inna_message_embeddings(chat_id);
create index if not exists inna_message_embeddings_chunk_idx on inna_message_embeddings(message_id, chunk_index);
create index if not exists inna_message_embeddings_section_idx on inna_message_embeddings(section_title);

-- Vector similarity search index
create index if not exists inna_message_embeddings_embedding_idx 
on inna_message_embeddings 
using ivfflat (embedding vector_cosine_ops)
with (lists = 100); 