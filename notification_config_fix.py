
# notification_config_fix.py
"""
Configuration fix to disable all notifications
==============================================
Import this to safely disable notification systems
"""

import logging
logger = logging.getLogger(__name__)

class DisabledNotificationManager:
    """Disabled notification manager - prevents errors"""
    
    def __init__(self, *args, **kwargs):
        logger.info("Notifications disabled - using safe fallback")
    
    async def send_notification(self, *args, **kwargs):
        logger.debug("Notification disabled (would have sent)")
        return {"status": "disabled", "message": "Notifications are disabled"}
    
    async def send_sms(self, *args, **kwargs):
        logger.debug("SMS disabled (would have sent)")
        return {"status": "disabled", "message": "SMS notifications are disabled"}
    
    async def send_email(self, *args, **kwargs):
        logger.debug("Email disabled (would have sent)")
        return {"status": "disabled", "message": "Email notifications are disabled"}
    
    async def send_discord(self, *args, **kwargs):
        logger.debug("Discord disabled (would have sent)")
        return {"status": "disabled", "message": "Discord notifications are disabled"}
    
    async def send_slack(self, *args, **kwargs):
        logger.debug("Slack disabled (would have sent)")
        return {"status": "disabled", "message": "Slack notifications are disabled"}

# Safe imports that won't cause errors
try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    class TwilioClient:
        def __init__(self, *args, **kwargs):
            logger.warning("Twilio not available - using fallback")
        
        @property
        def messages(self):
            return self
        
        def create(self, *args, **kwargs):
            return {"sid": "disabled", "status": "not_sent"}

# Export safe versions
__all__ = ['DisabledNotificationManager', 'TwilioClient']
