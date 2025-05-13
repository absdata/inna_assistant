from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from services.database import db_service
import json

app = FastAPI(title="Inna AI Dashboard")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Pydantic models
class Task(BaseModel):
    id: int
    title: str
    description: Optional[str]
    status: str
    priority: int
    due_date: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    chat_id: int
    assigned_to: Optional[str]

class Summary(BaseModel):
    id: int
    chat_id: int
    summary_type: str
    content: str
    period_start: datetime
    period_end: datetime
    created_at: datetime
    gdoc_url: Optional[str]

class Stats(BaseModel):
    total_messages: int
    total_tasks: int
    completed_tasks: int
    pending_tasks: int
    total_summaries: int

# Routes
@app.get("/")
async def root():
    """Serve the dashboard HTML."""
    with open("web/static/index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/api/tasks/{chat_id}")
async def get_tasks(chat_id: int) -> List[Task]:
    """Get all tasks for a chat."""
    tasks = await db_service.get_tasks(chat_id)
    return [Task(**task) for task in tasks]

@app.get("/api/summaries/{chat_id}")
async def get_summaries(chat_id: int) -> List[Summary]:
    """Get all summaries for a chat."""
    summaries = await db_service.get_summaries(chat_id)
    return [Summary(**summary) for summary in summaries]

@app.get("/api/stats/{chat_id}")
async def get_stats(chat_id: int) -> Stats:
    """Get statistics for a chat."""
    try:
        total_messages = await db_service.count_messages(chat_id)
        tasks = await db_service.get_tasks(chat_id)
        total_tasks = len(tasks)
        completed_tasks = sum(1 for task in tasks if task['status'] == 'completed')
        pending_tasks = total_tasks - completed_tasks
        total_summaries = await db_service.count_summaries(chat_id)
        
        return Stats(
            total_messages=total_messages,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            pending_tasks=pending_tasks,
            total_summaries=total_summaries
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/active_chats")
async def get_active_chats() -> List[int]:
    """Get all active chat IDs."""
    chats = await db_service.get_active_chats()
    return [chat['chat_id'] for chat in chats] 