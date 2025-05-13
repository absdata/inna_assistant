-- Create indexes for memory and summary tables
create index if not exists inna_summaries_chat_id_idx on inna_summaries(chat_id);
create index if not exists inna_agent_memory_chat_id_idx on inna_agent_memory(chat_id);
create index if not exists inna_agent_memory_role_idx on inna_agent_memory(agent_role);
create index if not exists inna_agent_memory_created_at_idx on inna_agent_memory(created_at); 