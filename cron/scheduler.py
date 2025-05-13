import asyncio
from datetime import datetime, timedelta
from services.database import db_service
from agent.summary import summary_agent
import logging

logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self):
        self.running = False
        self.tasks = []
    
    async def generate_summaries(self):
        """Generate weekly and monthly summaries for all active chats."""
        try:
            # Get all active chat IDs
            chats = await db_service.get_active_chats()
            
            for chat in chats:
                chat_id = chat['chat_id']
                
                # Check if we need a weekly summary
                last_weekly = await db_service.get_latest_summary(
                    chat_id=chat_id,
                    summary_type="weekly"
                )
                
                if not last_weekly or datetime.utcnow() - last_weekly['created_at'] >= timedelta(days=7):
                    logger.info(f"Generating weekly summary for chat {chat_id}")
                    await summary_agent.generate_weekly_summary(chat_id)
                
                # Check if we need a monthly summary
                last_monthly = await db_service.get_latest_summary(
                    chat_id=chat_id,
                    summary_type="monthly"
                )
                
                if not last_monthly or datetime.utcnow() - last_monthly['created_at'] >= timedelta(days=30):
                    logger.info(f"Generating monthly summary for chat {chat_id}")
                    await summary_agent.generate_monthly_summary(chat_id)
        
        except Exception as e:
            logger.error(f"Error generating summaries: {e}")
    
    async def _run_periodic(self, interval: int, func, *args, **kwargs):
        """Run a function periodically."""
        while self.running:
            try:
                await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in periodic task {func.__name__}: {e}")
            await asyncio.sleep(interval)
    
    def start(self):
        """Start the scheduler."""
        self.running = True
        
        # Schedule weekly summaries (check every hour)
        self.tasks.append(
            asyncio.create_task(
                self._run_periodic(3600, self.generate_summaries)
            )
        )
        
        logger.info("Scheduler started")
    
    async def stop(self):
        """Stop the scheduler."""
        self.running = False
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        logger.info("Scheduler stopped")

scheduler = Scheduler() 