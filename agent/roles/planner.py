from typing import List, Dict, Any
from .base import BaseAgent, AgentMemory
from services.azure_openai import openai_service
from services.database import db_service
import logging

logger = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__(role="planner")
    
    async def process(self, memory: AgentMemory) -> str:
        """Process the current state and create/update tasks."""
        # Get relevant context
        context = await self.get_context(
            memory.current_message.get("text", ""),
            memory.chat_id
        )
        
        # Format context for the LLM
        context_sections = []
        for item in memory.context:
            if item["type"] == "document":
                doc_text = f"Document: {item['file_name']} (Relevance: {item['relevance']:.2f}):\n"
                for section in item["content"]:
                    doc_text += f"Section (Similarity: {section['similarity']:.2f}):\n{section['content']}\n"
                context_sections.append(doc_text)
            elif item["type"] == "summary":
                context_sections.append(
                    f"Summary (Relevance: {item['relevance']:.2f}):\n{item['content']}"
                )
            elif item["type"] == "agent_insight":
                context_sections.append(
                    f"Agent {item['insight_type'].title()} (Relevance: {item['relevance']:.2f}):\n{item['content']}"
                )
            elif item["type"] == "chat":
                context_sections.append(
                    f"Message (Relevance: {item['relevance']:.2f}):\n{item['content']}"
                )
        
        context_text = "\n\n".join(context_sections)
        
        # Get existing tasks
        tasks = await db_service.get_tasks(memory.chat_id)
        tasks_text = "\n".join([
            f"- {task['title']} ({task['status']})" for task in tasks
        ])
        
        # Create messages for task planning
        messages = [
            openai_service.create_system_message(
                "You are a strategic planner for a startup. Your role is to create and manage tasks "
                "based on conversations and previous context. Break down big goals into smaller, "
                "actionable tasks and maintain a clear roadmap."
            ),
            openai_service.create_user_message(
                f"Context:\n{context_text}\n\n"
                f"Current Tasks:\n{tasks_text}\n\n"
                f"User Message: {memory.current_message.get('text', '')}\n\n"
                "Based on this information, what tasks should be created, updated, or completed? "
                "Format your response as a list of actions to take."
            )
        ]
        
        # Get plan from LLM
        plan = await openai_service.get_completion(messages)
        
        # Save the planning session to memory
        await self.save_memory(
            chat_id=memory.chat_id,
            context=f"Planning Session:\nInput: {memory.current_message.get('text', '')}\nPlan: {plan}"
        )
        
        # Parse and execute the plan
        try:
            # Create messages for task creation/updates
            messages = [
                openai_service.create_system_message(
                    "You are a task parser. Convert the planning output into a structured list of "
                    "task operations. Each operation should be either CREATE or UPDATE, followed by "
                    "the task details in a clear format."
                ),
                openai_service.create_user_message(f"Plan:\n{plan}")
            ]
            
            task_operations = await openai_service.get_completion(messages)
            
            # Execute task operations
            for operation in task_operations.split("\n"):
                operation = operation.strip()
                if not operation:
                    continue
                    
                if operation.startswith("CREATE:"):
                    task_details = operation[8:].strip()
                    if task_details:
                        await db_service.create_task(
                            chat_id=memory.chat_id,
                            title=task_details
                        )
                elif operation.startswith("UPDATE:"):
                    task_details = operation[8:].strip()
                    try:
                        # More robust parsing with error handling
                        if " -> " not in task_details:
                            logger.warning(f"Invalid task update format: {task_details}")
                            continue
                            
                        task_id_str, new_status = task_details.split(" -> ", 1)
                        task_id = int(task_id_str.strip())
                        new_status = new_status.strip()
                        
                        if task_id > 0 and new_status:
                            await db_service.update_task(
                                task_id=task_id,
                                status=new_status
                            )
                        else:
                            logger.warning(f"Invalid task ID or status: ID={task_id}, status={new_status}")
                    except ValueError as e:
                        logger.error(f"Error parsing task update '{task_details}': {str(e)}")
                    except Exception as e:
                        logger.error(f"Error updating task: {str(e)}")
        
        except Exception as e:
            print(f"Error executing task operations: {e}")
        
        return plan

planner_agent = PlannerAgent() 