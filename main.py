# Entry point for the Inna bot
import asyncio
import signal
import uvicorn
from services.telegram_bot import bot_service
from cron.scheduler import scheduler
from web.app import app
import logging
import sys
from contextlib import asynccontextmanager

# Set up detailed logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,  # Change to DEBUG level for more detailed logs
    stream=sys.stdout  # Ensure logs go to stdout
)

# Create logger for this module
logger = logging.getLogger(__name__)

# Enable debug logs for key components
logging.getLogger('telegram').setLevel(logging.DEBUG)
logging.getLogger('aiohttp').setLevel(logging.DEBUG)
logging.getLogger('uvicorn').setLevel(logging.INFO)

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
        bot_task = asyncio.create_task(bot_service.start())
        logger.info("Telegram bot service started in background")
        
        logger.info("All services started successfully")
        yield
        
        logger.info("=== Beginning shutdown sequence ===")
        
        # Stop the scheduler
        logger.info("Stopping scheduler service...")
        await scheduler.stop()
        logger.info("Scheduler service stopped successfully")
        
        # Stop the Telegram bot
        logger.info("Stopping Telegram bot service...")
        await bot_service.stop()
        
        # Wait for bot task to complete
        if not bot_task.done():
            logger.info("Waiting for bot task to complete...")
            try:
                await asyncio.wait_for(bot_task, timeout=5.0)
                logger.info("Bot task completed successfully")
            except asyncio.TimeoutError:
                logger.warning("Bot task timeout, forcing cancellation...")
                bot_task.cancel()
                try:
                    await bot_task
                except asyncio.CancelledError:
                    logger.info("Bot task cancelled")
        
        logger.info("All services stopped successfully")
        
    except Exception as e:
        logger.error(f"Critical error in lifespan context: {str(e)}", exc_info=True)
        raise

app.router.lifespan_context = lifespan

async def main():
    """Main function to run both the bot and web dashboard."""
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="debug",  # Change to debug for more detailed logs
        access_log=True
    )
    server = uvicorn.Server(config)
    
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(server)))
        logger.info(f"Registered signal handler for {sig.name}")
    
    try:
        logger.info("=== Inna AI Assistant Starting ===")
        logger.info("Starting web server...")
        await server.serve()
    except Exception as e:
        logger.error(f"Critical error running services: {str(e)}", exc_info=True)
        await shutdown(server)

async def shutdown(server):
    """Gracefully shut down all services."""
    try:
        logger.info("=== Beginning emergency shutdown ===")
        logger.info("Shutting down web server...")
        await server.shutdown()
        logger.info("Web server shutdown complete")
    except Exception as e:
        logger.error(f"Error during emergency shutdown: {str(e)}", exc_info=True)
    finally:
        logger.info("Stopping event loop...")
        loop = asyncio.get_running_loop()
        loop.stop()
        logger.info("Event loop stopped")

if __name__ == "__main__":
    try:
        logger.info("=== Initializing Inna AI Assistant ===")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Services stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)