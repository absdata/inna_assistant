# Entry point for the Inna bot
import asyncio
import signal
import uvicorn
from services.telegram_bot import telegram_bot
from cron.scheduler import scheduler
from web.app import app
import logging
import sys
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Set up detailed logging
log_filename = logs_dir / f"inna_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging format and handlers
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
    handlers=[
        # Console handler
        logging.StreamHandler(sys.stdout),
        # File handler
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)

# Create logger for this module
logger = logging.getLogger(__name__)

# Enable debug logs for key components
logging.getLogger('telegram').setLevel(logging.DEBUG)
logging.getLogger('aiohttp').setLevel(logging.DEBUG)
logging.getLogger('uvicorn').setLevel(logging.INFO)

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