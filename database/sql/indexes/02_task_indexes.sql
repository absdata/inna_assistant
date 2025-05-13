-- Create indexes for task-related tables
create index if not exists inna_tasks_chat_id_idx on inna_tasks(chat_id);
create index if not exists inna_tasks_status_idx on inna_tasks(status);
create index if not exists inna_tasks_due_date_idx on inna_tasks(due_date);
create index if not exists inna_tasks_priority_idx on inna_tasks(priority); 