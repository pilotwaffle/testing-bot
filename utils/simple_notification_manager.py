"""
Simple Notification Manager
==========================
"""

import logging
from typing import Optional

class SimpleNotificationManager:
    """Simple notification manager that logs notifications"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Simple Notification Manager initialized")
    
    async def send_notification(self, title: str, message: str, priority: str = "INFO"):
        """Send notification (logs only in demo mode)"""
        self.logger.info(f"[{priority}] {title}: {message}")
        return True
    
    async def notify(self, title: str, message: str, priority: str = "INFO"):
        """Alias for send_notification"""
        return await self.send_notification(title, message, priority)
    
    def is_enabled(self) -> bool:
        """Check if notifications are enabled"""
        return False  # Disabled by default to avoid issues
