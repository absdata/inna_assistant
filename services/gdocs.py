from google.oauth2.credentials import Credentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import Optional, Dict, Any
import json
from config.config import config
from services.database import db_service

class GoogleDocsService:
    def __init__(self):
        self.credentials = ServiceAccountCredentials.from_service_account_info(
            json.loads(config.google_service_account)
        )
        self.docs_service = build('docs', 'v1', credentials=self.credentials)
        self.drive_service = build('drive', 'v3', credentials=self.credentials)
    
    async def create_doc(self, title: str, content: str) -> Optional[str]:
        """Create a new Google Doc and return its ID."""
        try:
            # Create an empty document
            doc_metadata = {
                'title': title,
                'mimeType': 'application/vnd.google-apps.document'
            }
            doc = self.drive_service.files().create(
                body=doc_metadata,
                fields='id'
            ).execute()
            
            # Update document content
            requests = [
                {
                    'insertText': {
                        'location': {'index': 1},
                        'text': content
                    }
                }
            ]
            
            self.docs_service.documents().batchUpdate(
                documentId=doc.get('id'),
                body={'requests': requests}
            ).execute()
            
            return doc.get('id')
        
        except HttpError as error:
            print(f"Error creating Google Doc: {error}")
            return None
    
    async def update_doc(self, doc_id: str, content: str) -> bool:
        """Update an existing Google Doc with new content."""
        try:
            # Get the current document content
            document = self.docs_service.documents().get(documentId=doc_id).execute()
            
            # Create a request to delete all content
            requests = [
                {
                    'deleteContentRange': {
                        'range': {
                            'startIndex': 1,
                            'endIndex': document.get('body').get('content')[-1].get('endIndex') - 1
                        }
                    }
                },
                {
                    'insertText': {
                        'location': {'index': 1},
                        'text': content
                    }
                }
            ]
            
            self.docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={'requests': requests}
            ).execute()
            
            return True
        
        except HttpError as error:
            print(f"Error updating Google Doc: {error}")
            return False
    
    async def sync_summary(self, summary_id: int) -> bool:
        """Sync a summary with Google Docs."""
        try:
            # Get the summary from database
            summary = await db_service.get_summary(summary_id)
            if not summary:
                return False
            
            # Check if a Google Doc already exists
            sync_record = await db_service.get_gdoc_sync(
                reference_id=summary_id,
                doc_type='summary'
            )
            
            if sync_record and sync_record.get('doc_id'):
                # Update existing doc
                success = await self.update_doc(
                    sync_record['doc_id'],
                    summary['content']
                )
            else:
                # Create new doc
                title = f"Summary: {summary['summary_type']} ({summary['period_start']} - {summary['period_end']})"
                doc_id = await self.create_doc(title, summary['content'])
                if doc_id:
                    # Save sync record
                    await db_service.save_gdoc_sync(
                        doc_id=doc_id,
                        doc_type='summary',
                        reference_id=summary_id
                    )
                    success = True
                else:
                    success = False
            
            return success
        
        except Exception as e:
            print(f"Error syncing summary to Google Docs: {e}")
            return False
    
    async def sync_tasks(self, chat_id: int) -> bool:
        """Sync tasks with Google Docs."""
        try:
            # Get all tasks for the chat
            tasks = await db_service.get_tasks(chat_id)
            
            # Format tasks content
            content = "# Task List\n\n"
            for task in tasks:
                status_emoji = "✅" if task['status'] == 'completed' else "⏳"
                due_date = f"Due: {task['due_date']}" if task['due_date'] else "No due date"
                content += f"{status_emoji} {task['title']}\n"
                content += f"Priority: {'❗' * task['priority']}\n"
                content += f"{due_date}\n"
                if task['description']:
                    content += f"Description: {task['description']}\n"
                content += "\n"
            
            # Check if a Google Doc already exists
            sync_record = await db_service.get_gdoc_sync(
                reference_id=chat_id,
                doc_type='tasks'
            )
            
            if sync_record and sync_record.get('doc_id'):
                # Update existing doc
                success = await self.update_doc(
                    sync_record['doc_id'],
                    content
                )
            else:
                # Create new doc
                title = f"Tasks List - Chat {chat_id}"
                doc_id = await self.create_doc(title, content)
                if doc_id:
                    # Save sync record
                    await db_service.save_gdoc_sync(
                        doc_id=doc_id,
                        doc_type='tasks',
                        reference_id=chat_id
                    )
                    success = True
                else:
                    success = False
            
            return success
        
        except Exception as e:
            print(f"Error syncing tasks to Google Docs: {e}")
            return False

gdocs_service = GoogleDocsService() 