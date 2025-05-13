from typing import List, Dict, Any
from datetime import datetime, timedelta
from services.azure_openai import openai_service
from services.database import db_service
from services.gdocs import gdocs_service

class SummaryAgent:
    @staticmethod
    async def generate_weekly_summary(chat_id: int) -> Dict[str, Any]:
        """Generate a weekly summary of activities and updates."""
        # Calculate the date range for the past week
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        # Get messages from the past week
        messages = await db_service.get_chat_history_by_date_range(
            chat_id=chat_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Get tasks updated in the past week
        tasks = await db_service.get_tasks_by_date_range(
            chat_id=chat_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Format the content for the LLM
        messages_text = "\n".join([
            f"- {msg['text']}" for msg in messages if msg.get('text')
        ])
        
        tasks_text = "\n".join([
            f"- {task['title']} ({task['status']})" for task in tasks
        ])
        
        # Create messages for summary generation
        llm_messages = [
            openai_service.create_system_message(
                "You are a business analyst creating weekly summaries for a startup. "
                "Analyze the week's activities and create a structured summary that includes:\n"
                "1. Key Discussions & Decisions\n"
                "2. Progress on Tasks\n"
                "3. Important Updates\n"
                "4. Action Items for Next Week"
            ),
            openai_service.create_user_message(
                f"Time Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n"
                f"Messages:\n{messages_text}\n\n"
                f"Tasks:\n{tasks_text}\n\n"
                "Please create a comprehensive weekly summary."
            )
        ]
        
        # Generate the summary
        summary_content = await openai_service.get_completion(llm_messages)
        
        # Save the summary to database
        summary = await db_service.save_summary(
            chat_id=chat_id,
            summary_type="weekly",
            content=summary_content,
            period_start=start_date,
            period_end=end_date
        )
        
        # Sync with Google Docs
        if summary:
            await gdocs_service.sync_summary(summary["id"])
        
        return summary

    @staticmethod
    async def generate_monthly_summary(chat_id: int) -> Dict[str, Any]:
        """Generate a monthly summary of activities and updates."""
        # Calculate the date range for the past month
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        # Get weekly summaries for the month
        weekly_summaries = await db_service.get_summaries_by_date_range(
            chat_id=chat_id,
            summary_type="weekly",
            start_date=start_date,
            end_date=end_date
        )
        
        # Format the content for the LLM
        summaries_text = "\n\n".join([
            f"Week of {summary['period_start'].strftime('%Y-%m-%d')}:\n{summary['content']}"
            for summary in weekly_summaries
        ])
        
        # Create messages for monthly summary generation
        llm_messages = [
            openai_service.create_system_message(
                "You are a business analyst creating monthly summaries for a startup. "
                "Review the weekly summaries and create a comprehensive monthly report that includes:\n"
                "1. Executive Summary\n"
                "2. Major Milestones & Achievements\n"
                "3. Key Challenges & Solutions\n"
                "4. Progress Towards Goals\n"
                "5. Strategic Recommendations"
            ),
            openai_service.create_user_message(
                f"Time Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n"
                f"Weekly Summaries:\n{summaries_text}\n\n"
                "Please create a comprehensive monthly summary."
            )
        ]
        
        # Generate the summary
        summary_content = await openai_service.get_completion(llm_messages)
        
        # Save the summary to database
        summary = await db_service.save_summary(
            chat_id=chat_id,
            summary_type="monthly",
            content=summary_content,
            period_start=start_date,
            period_end=end_date
        )
        
        # Sync with Google Docs
        if summary:
            await gdocs_service.sync_summary(summary["id"])
        
        return summary

summary_agent = SummaryAgent() 