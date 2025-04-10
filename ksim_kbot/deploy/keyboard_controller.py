"""Asynchronous keyboard input."""

import asyncio
import logging
import select
import sys
import termios
from contextlib import contextmanager
from typing import Awaitable, Callable, Generator, Optional

logger = logging.getLogger(__name__)


class KeyboardController:
    """Handles asynchronous keyboard input in a non-blocking way."""

    def __init__(self, key_handler: Callable[[str], Awaitable[None]]) -> None:
        """Initializes the KeyboardController.

        Args:
            key_handler: An async function to call when a key is pressed.
                         It receives the pressed key (str) as an argument.
        """
        self._key_handler = key_handler
        self._task: Optional[asyncio.Task] = None

    @contextmanager
    def _cbreak(self) -> Generator[None, None, None]:
        """Context manager for terminal cbreak mode - allows char-by-char input without echo."""
        try:
            # Save original terminal settings
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)

            # Configure terminal for cbreak mode
            new_settings = termios.tcgetattr(fd)
            new_settings[3] = new_settings[3] & ~termios.ECHO  # Disable echo
            new_settings[3] = new_settings[3] & ~termios.ICANON  # Disable canonical mode
            new_settings[6][termios.VMIN] = 0  # No blocking
            new_settings[6][termios.VTIME] = 0  # No timeout
            termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)

            yield
        finally:
            # Restore terminal settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    async def _run_loop(self) -> None:
        """The main loop that listens for keyboard input."""
        logger.info("\nKeyboard control active:")
        logger.info("  w/s: forward/backward")
        logger.info("  a/d: left/right")
        logger.info("  q/e: down/up")
        logger.info("  Press Ctrl+C to exit\n")

        with self._cbreak():
            try:
                while True:
                    # Check if input is available
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key == "\x03":  # Ctrl+C
                            logger.info("Ctrl+C detected by keyboard listener.")
                            # Request main loop cancellation (handled by KeyboardInterrupt propagation)
                            current_task = asyncio.current_task()
                            if current_task:
                                current_task.get_loop().call_soon(current_task.get_loop().stop)
                            break

                        # Call the provided handler
                        await self._key_handler(key)

                    # Yield to other tasks
                    await asyncio.sleep(0.01)
            except (asyncio.CancelledError):
                 logger.debug("Keyboard listener task cancelled.")
            except Exception as e:
                logger.exception("Error in keyboard listener loop: %s", e)
            finally:
                logger.info("Keyboard input loop stopping...")


    async def start(self) -> None:
        """Starts the keyboard listening task."""
        if self._task is None or self._task.done():
            logger.info("Starting keyboard listener...")
            self._task = asyncio.create_task(self._run_loop())
            self._task.add_done_callback(self._task_done_callback)
        else:
            logger.warning("Keyboard listener task already running.")

    def _task_done_callback(self, task: asyncio.Task) -> None:
        """Callback executed when the keyboard task finishes."""
        try:
            # Check if the task raised an exception
            exc = task.exception()
            if exc:
                logger.exception("Keyboard listener task failed: %s", exc)
            else:
                logger.info("Keyboard listener task finished successfully.")
        except asyncio.CancelledError:
             logger.info("Keyboard listener task was cancelled.")
        self._task = None # Clear the task reference

    async def stop(self) -> None:
        """Stops the keyboard listening task."""
        if self._task and not self._task.done():
            logger.info("Stopping keyboard listener task...")
            self._task.cancel()
            try:
                # Wait briefly for the task to acknowledge cancellation
                await asyncio.wait_for(self._task, timeout=1.0)
            except asyncio.CancelledError:
                logger.debug("Keyboard listener task successfully cancelled.")
            except asyncio.TimeoutError:
                logger.warning("Keyboard listener task did not stop gracefully.")
            except Exception as e:
                logger.exception("Error stopping keyboard listener task: %s", e)
            finally:
                self._task = None
        else:
            logger.debug("Keyboard listener task not running or already stopped.") 
