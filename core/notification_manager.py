"""
Simple Notification Manager - Fixed Version
==========================================
"""

import logging
from typing import Optional, Dict, Any

class SimpleNotificationManager:
    """Simple notification manager that just logs (no external services)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Simple Notification Manager initialized (notifications disabled)")
    
    async def send_notification(self, title: str, message: str, priority: str = "INFO"):
        """Send notification (logs only)"""
        self.logger.info(f"[{priority}] {title}: {message}")
        return {"status": "logged", "message": "Notification logged successfully"}
    
    async def notify(self, title: str, message: str, priority: str = "INFO"):
        """Alias for send_notification"""
        return await self.send_notification(title, message, priority)
    
    async def send_sms(self, *args, **kwargs):
        """SMS disabled"""
        self.logger.debug("SMS notification disabled")
        return {"status": "disabled", "message": "SMS notifications are disabled"}
    
    async def send_email(self, *args, **kwargs):
        """Email disabled"""
        self.logger.debug("Email notification disabled")
        return {"status": "disabled", "message": "Email notifications are disabled"}
    
    async def send_discord(self, *args, **kwargs):
        """Discord disabled"""
        self.logger.debug("Discord notification disabled")
        return {"status": "disabled", "message": "Discord notifications are disabled"}
    
    async def send_slack(self, *args, **kwargs):
        """Slack disabled"""
        self.logger.debug("Slack notification disabled")
        return {"status": "disabled", "message": "Slack notifications are disabled"}
    
    def is_enabled(self) -> bool:
        """Check if notifications are enabled"""
        return False  # All external notifications disabled
    
    def get_status(self) -> Dict[str, Any]:
        """Get notification system status"""
        return {
            "enabled": False,
            "services": {
                "sms": False,
                "email": False,
                "discord": False,
                "slack": False
            },
            "mode": "logging_only"
        }

# Backward compatibility
NotificationManager = SimpleNotificationManager

# Additional safe classes for any other imports
class DisabledEmailService:
    def __init__(self, *args, **kwargs):
        pass
    
    def send(self, *args, **kwargs):
        return {"status": "disabled"}

class DisabledSMSService:
    def __init__(self, *args, **kwargs):
        pass
    
    def send(self, *args, **kwargs):
        return {"status": "disabled"}

class DisabledWebhookService:
    def __init__(self, *args, **kwargs):
        pass
    
    def send(self, *args, **kwargs):
        return {"status": "disabled"}
