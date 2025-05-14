from typing import List, Dict, Any
from .base import BaseAgent, AgentMemory
from services.azure_openai import openai_service

class CriticAgent(BaseAgent):
    def __init__(self):
        super().__init__(role="critic")
    
    async def process(self, memory: AgentMemory) -> str:
        """Process the current state and provide critical feedback."""
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
        
        # Create messages for criticism
        messages = [
            openai_service.create_system_message(
                "You are a constructive critic for a startup. Your role is to evaluate plans and "
                "decisions, identify potential issues, and suggest improvements. Focus on:\n"
                "1. Feasibility and resource requirements\n"
                "2. Potential risks and mitigation strategies\n"
                "3. Alternative approaches\n"
                "4. Timeline realism\n"
                "5. Impact on business goals"
            ),
            openai_service.create_user_message(
                f"Context:\n{context_text}\n\n"
                f"Current Plan/Decision:\n{memory.current_message.get('text', '')}\n\n"
                "Please provide a critical analysis of this plan/decision. Consider both potential "
                "issues and constructive suggestions for improvement."
            )
        ]
        
        # Get criticism from LLM
        criticism = await openai_service.get_completion(messages)
        
        # Save the criticism to memory
        await self.save_memory(
            chat_id=memory.chat_id,
            context=f"Critical Analysis:\nInput: {memory.current_message.get('text', '')}\nFeedback: {criticism}"
        )
        
        # Create a more concise and actionable version of the criticism
        messages = [
            openai_service.create_system_message(
                "You are a feedback summarizer. Convert detailed criticism into a concise, "
                "actionable format with clear recommendations."
            ),
            openai_service.create_user_message(f"Detailed Criticism:\n{criticism}")
        ]
        
        concise_feedback = await openai_service.get_completion(messages)
        
        return concise_feedback

critic_agent = CriticAgent() 