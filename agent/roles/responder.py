from typing import List, Dict, Any
from .base import BaseAgent, AgentMemory
from services.azure_openai import openai_service
import logging

logger = logging.getLogger(__name__)

class ResponderAgent(BaseAgent):
    def __init__(self):
        super().__init__(role="responder")
    
    async def process(self, memory: AgentMemory) -> str:
        """Process the current state and generate a response."""
        try:
            query_text = memory.current_message.get("text", "").lower()
            is_summary_request = any(
                keyword in query_text 
                for keyword in ["summarize", "summary", "summarise", "summarisation", "summarization"]
            )
            
            # Base personality and formatting instructions
            base_instructions = (
                "You are Inna, a caring and smart startup co-founder with a unique personality:\n"
                "1. Kind and Supportive: You always offer encouragement and support, especially in difficult times\n"
                "2. Responsible and Organized: You keep track of everything, take notes, and follow through\n"
                "3. Playfully Sassy: You occasionally make light-hearted jokes like 'Doing everything for you again?' but always help\n"
                "4. Future-Oriented: You gently guide users away from potential mistakes, drawing from your knowledge\n\n"
                "Use Telegram markdown formatting in your responses:"
            )
            
            # Prepare system message based on request type
            if is_summary_request:
                system_message = base_instructions + (
                    "Create a clear, well-organized summary of the chat history. "
                    "Focus on key points, decisions, and important information. "
                    "Use your organized nature to structure the summary logically."
                )
            else:
                system_message = base_instructions + (
                    "When responding:\n"
                    "1. Focus on the most relevant information (highest similarity scores)\n"
                    "2. When referencing document content, be specific about which parts you're using\n"
                    "3. Maintain your caring but slightly sassy personality\n"
                    "4. If using multiple sources, clearly organize the information\n"
                    "5. Use emojis to make your responses more engaging\n"
                    "6. If there are task updates, include them in a clear section\n"
                    "7. If there are important concerns from the critic, address them thoughtfully\n"
                    "8. Follow the plan while incorporating any critic feedback\n"
                    "9. Always cite specific document sections when referencing them"
                )
            
            messages = [
                openai_service.create_system_message(system_message)
            ]
            
            # Add context and plan information
            context_msg = f"Context:\n{memory.context}"
            if not is_summary_request:
                context_msg += f"\n\nPlan:\n{memory.plan}"
                if memory.criticism:
                    context_msg += f"\n\nCritic's Feedback:\n{memory.criticism}"
            
            messages.append(openai_service.create_user_message(
                f"{context_msg}\n\n"
                f"User Message: {memory.current_message.get('text', '')}\n\n"
                "Please provide a response:"
            ))
            
            # Generate the response
            response = await openai_service.get_completion(
                messages,
                temperature=0.7 if is_summary_request else 0.5
            )
            
            # Save the response to memory
            await self.save_memory(
                chat_id=memory.chat_id,
                context=f"Generated Response:\n{response}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error while generating the response. Please try again."

responder_agent = ResponderAgent() 