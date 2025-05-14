# Entry point for the Inna bot
import asyncio
import signal
import uvicorn
from services.telegram_bot import telegram_bot
from cron.scheduler import scheduler
from web.app import app
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

class ConsoleFormatter(logging.Formatter):
    """Custom formatter to exclude embedding fields from console output."""
    def format(self, record):
        # Create a copy of the record to avoid modifying the original
        if isinstance(record.args, dict):
            filtered_args = {k: v for k, v in record.args.items() if 'embedding' not in k.lower()}
            record.args = filtered_args
        elif isinstance(record.msg, dict):
            filtered_msg = {k: v for k, v in record.msg.items() if 'embedding' not in k.lower()}
            record.msg = filtered_msg
        elif isinstance(record.msg, str):
            # Remove embedding-related content from string messages
            msg_lines = record.msg.split('\n')
            filtered_lines = [line for line in msg_lines if 'embedding' not in line.lower()]
            record.msg = '\n'.join(filtered_lines)
        
        return super().format(record)

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True, parents=True)

# Set up detailed logging
log_filename = logs_dir / f"inna_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Ensure the log file can be created
try:
    log_filename.touch(exist_ok=True)
except Exception as e:
    print(f"Error creating log file: {e}")
    sys.exit(1)

# Create formatters
console_formatter = ConsoleFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(console_formatter)
console_handler.setLevel(logging.INFO)  # Console shows INFO and above

file_handler = logging.FileHandler(log_filename, encoding='utf-8', mode='a')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)  # File shows all DEBUG and above

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # Allow all logs to be processed
root_logger.handlers = []  # Remove any existing handlers
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Create logger for this module
logger = logging.getLogger(__name__)

# Configure third-party loggers
logging.getLogger('telegram').setLevel(logging.INFO)
logging.getLogger('aiohttp').setLevel(logging.INFO)
logging.getLogger('uvicorn').setLevel(logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

logger.info(f"Logging to file: {log_filename}")

@asynccontextmanager
async def lifespan(app):
    """Startup and shutdown events for FastAPI."""
    try:
        logger.info("=== Starting Inna AI Assistant Services ===")
        
        # Start the scheduler
        logger.info("Starting scheduler service...")
        scheduler.start()
        logger.info("Scheduler service started successfully")
        
        # Start the Telegram bot in the background
        logger.info("Starting Telegram bot service...")
        bot_task = asyncio.create_task(telegram_bot.start())
        logger.info("Telegram bot service started in background")
        
        logger.info("All services started successfully")
        yield
        
        # Begin shutdown sequence
        logger.info("=== Beginning shutdown sequence ===")
        
        # Stop the scheduler
        logger.info("Stopping scheduler service...")
        await scheduler.stop()
        logger.info("Scheduler service stopped successfully")
        
        # Stop the Telegram bot
        logger.info("Stopping Telegram bot service...")
        await telegram_bot.stop()
        
        # Wait for bot task to complete with timeout
        if not bot_task.done():
            logger.info("Waiting for bot task to complete...")
            try:
                await asyncio.wait_for(bot_task, timeout=10.0)
                logger.info("Bot task completed successfully")
            except asyncio.TimeoutError:
                logger.warning("Bot task timeout, forcing cancellation...")
                bot_task.cancel()
                try:
                    await asyncio.wait_for(bot_task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.warning("Bot task cancelled")
        
        logger.info("All services stopped successfully")
        
    except Exception as e:
        logger.error(f"Critical error in lifespan context: {str(e)}", exc_info=True)
        # Ensure we attempt to stop services even if there was an error
        await shutdown(None)
        raise

app.router.lifespan_context = lifespan

async def main():
    """Main function to run both the bot and web dashboard."""
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="debug",
        access_log=True
    )
    server = uvicorn.Server(config)
    
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=server: asyncio.create_task(shutdown(s)))
        logger.info(f"Registered signal handler for {sig.name}")
    
    try:
        logger.info("=== Inna AI Assistant Starting ===")
        logger.info("Starting web server...")
        await server.serve()
    except Exception as e:
        logger.error(f"Critical error running services: {str(e)}", exc_info=True)
        await shutdown(server)
        raise

async def shutdown(server):
    """Graceful shutdown of all services."""
    logger.info("=== Beginning shutdown sequence ===")
    
    # Stop the scheduler first
    logger.info("Stopping scheduler service...")
    try:
        await scheduler.stop()
        logger.info("Scheduler service stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping scheduler: {str(e)}", exc_info=True)
    
    # Stop the Telegram bot
    logger.info("Stopping Telegram bot service...")
    try:
        await telegram_bot.stop()
        logger.info("Telegram bot service stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping Telegram bot: {str(e)}", exc_info=True)
    
    # Stop the server
    logger.info("Stopping web server...")
    try:
        await server.shutdown()
        logger.info("Web server stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping web server: {str(e)}", exc_info=True)
    
    logger.info("=== Shutdown sequence completed ===")

if __name__ == "__main__":
    try:
        logger.info("=== Initializing Inna AI Assistant ===")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Services stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)