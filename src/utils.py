"""
This file contains useful utility functions for the project.

"""

from datetime import datetime


class SimpleLogger:
    """
    A simple logger class that provides methods to log messages with a prefix and timestamp.
    """

    def __init__(self, prefix="LOG"):
        # Initializes the logger with a prefix and sets the initial time.
        self.prefix = prefix
        self.time = datetime.now()

    def log(self, message):
        """
        Logs a message with the current timestamp and the specified prefix.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{self.prefix} {timestamp}] {message}", flush=True)

    def info(self, message):
        """
        Logs an informational message.
        """
        self.log(f"INFO: {message}")

    def warn(self, message):
        # Logs a warning message.
        self.log(f"WARNING: {message}")

    def error(self, message):
        # Logs an error message.
        self.log(f"ERROR: {message}")

    def start_timer(self, timer_name):
        # Starts a timer and logs the start time with the given timer name.
        self.time = datetime.now()
        self.log(f"{timer_name}: Timer started.")

    def stop_timer(self, timer_name):
        # Stops the timer, calculates the elapsed time in minutes, and logs it with the timer name.
        elapsed_time = datetime.now() - self.time
        elapsed_time = elapsed_time.total_seconds()
        elapsed_time /= 60
        elapsed_time = round(elapsed_time, 2)
        self.log(f"{timer_name}: Timer stopped. Elapsed time: {elapsed_time} minutes.")
