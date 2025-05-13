# Entry point for the Inna bot
import asyncio
import signal
import uvicorn
from services.telegram_bot import bot_service
from cron.scheduler import scheduler
from web.app import app
import logging
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    """Startup and shutdown events for FastAPI."""
    # Start the scheduler
    scheduler.start()
    
    # Start the Telegram bot
    await bot_service.start()
    
    yield
    
    # Stop the scheduler
    await scheduler.stop()
    
    # Stop the Telegram bot
    await bot_service.stop()

app.router.lifespan_context = lifespan

async def main():
    """Main function to run both the bot and web dashboard."""
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(server)))
    
    try:
        logger.info("Starting Inna AI Assistant...")
        await server.serve()
    except Exception as e:
        logger.error(f"Error running services: {e}")
        await shutdown(server)

async def shutdown(server):
    """Gracefully shut down all services."""
    try:
        logger.info("Shutting down Inna AI Assistant...")
        await server.shutdown()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    finally:
        asyncio.get_event_loop().stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Services stopped by user")