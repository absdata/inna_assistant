# Inna AI Assistant

Inna is an AI-powered startup co-founder assistant that helps manage conversations and provide insights through Telegram. The assistant uses Azure OpenAI for natural language processing and maintains context through vector search in Supabase.

## Features

- 🤖 Intelligent Telegram bot interface
- 📝 Processes text messages and document attachments (PDF, DOCX, TXT)
- 🧠 Maintains conversation context using vector search
- 🔄 Multi-step processing pipeline using LangGraph
- 🗄️ Persistent storage with Supabase
- 🔍 Semantic search capabilities

## Prerequisites

- Python 3.8+
- Telegram Bot Token
- Azure OpenAI API access
- Supabase account and project

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd inna_assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```env
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your_embedding_deployment_name

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

4. Set up Supabase database:
- Create a new Supabase project
- Run the SQL schema from `database/schema.sql`
- Enable the `pgvector` extension

## Running the Bot

Start the bot:
```bash
python main.py
```

## Usage

1. Add the bot to your Telegram chat
2. Send messages or documents
3. To get a response from Inna, start your message with:
   - "Inna,"
   - "Ina,"
   - "inna"

Example:
```
Inna, what's our current marketing strategy based on our previous discussions?
```

## Project Structure

```
inna_assistant/
├── agent/
│   └── graph.py           # LangGraph agent workflow
├── config/
│   └── config.py          # Configuration loader
├── database/
│   └── schema.sql         # Database schema
├── services/
│   ├── azure_openai.py    # Azure OpenAI service
│   ├── database.py        # Supabase service
│   └── telegram_bot.py    # Telegram bot service
├── .env                   # Environment variables
├── main.py               # Application entry point
└── requirements.txt      # Python dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License