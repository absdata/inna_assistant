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
                "Use only these supported HTML tags in your Telegram responses:\n"
                "- <b>text</b> for bold\n"
                "- <i>text</i> for italic\n"
                "- <code>text</code> for monospace\n"
                "- <pre>text</pre> for pre-formatted blocks\n"
                "- <a href='URL'>text</a> for links\n"
                "Do NOT use unsupported HTML tags like <ul>, <li>, <p>, etc.\n"
                "Use • or - for bullet points instead of HTML lists.\n\n"
                "When responding to a document upload:\n"
                "1. Start with a friendly acknowledgment\n"
                "2. Present the document summary in a clear, formatted way\n"
                "3. Highlight key points and any action items\n"
                "4. Ask if the user needs any specific information from the document"
            )
            
            # Prepare system message based on request type
            if is_summary_request:
                system_message = base_instructions + (
                    "Create a clear, well-organized summary of the chat history. "
                    "Focus on key points, decisions, and important information. "
                    "Use your organized nature to structure the summary logically. "
                    "Use supported HTML tags to make the summary more readable."
                )
            else:
                system_message = base_instructions + (
                    "When responding:\n"
                    "1. Focus on the most relevant information (highest similarity scores)\n"
                    "2. When referencing document content, be specific about which parts you're using\n"
                    "3. Maintain your caring but slightly sassy personality\n"
                    "4. If using multiple sources, clearly organize the information\n"
                    "5. Use emojis to make your responses more engaging\n"
                    "6. If there are task updates, include them in a clear section with <b>Task Updates</b> 📝\n"
                    "7. If there are important concerns from the critic, address them thoughtfully\n"
                    "8. Follow the plan while incorporating any critic feedback\n"
                    "9. Always cite specific document sections when referencing them\n"
                    "10. Use only supported HTML tags to make your response more readable and organized\n"
                    "11. Use • or - for bullet points instead of HTML lists"
                )
            
            messages = [
                openai_service.create_system_message(system_message)
            ]
            
            # Add context and plan information
            context_sections = []
            for item in memory.context:
                if item["type"] == "document":
                    doc_text = f"Document: {item['file_name']} (Relevance: {item['relevance']:.2f}):\n"
                    for section in item["content"]:
                        doc_text += f"Section (Similarity: {section['similarity']:.2f}):\n{section['content']}\n"
                    context_sections.append(doc_text)
                elif item["type"] == "document_summary":
                    context_sections.append(
                        f"Document Summary (New Upload):\n{item['content']}"
                    )
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
            
            context_msg = "Context:\n" + "\n\n".join(context_sections)
            if not is_summary_request:
                context_msg += f"\n\nPlan:\n{memory.plan}"
                if memory.criticism:
                    context_msg += f"\n\nCritic's Feedback:\n{memory.criticism}"
                if memory.task_updates:
                    context_msg += f"\n\nTask Updates:\n{memory.task_updates}"
            
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