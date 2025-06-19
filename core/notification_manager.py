# core/notification_manager.py
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import deque # For notification history

from core.config import settings

logger = logging.getLogger(__name__)

# Define Notification Priority Levels
class NotificationPriority:
    DEBUG = 0
    INFO = 1
    # Adding WARNING and ERROR as explicit numeric levels
    WARNING = 2
    ERROR = 3 # <--- ADDED THIS LINE
    HIGH = 4 # Shifted HIGH from 3 to 4
    CRITICAL = 5 # Shifted CRITICAL from 4 to 5
    EMERGENCY = 6 # Shifted EMERGENCY from 5 to 6

    # Update STR_TO_LEVEL to reflect new numeric levels
    STR_TO_LEVEL = {
        "DEBUG": DEBUG, "INFO": INFO, "WARNING": WARNING, "ERROR": ERROR,
        "HIGH": HIGH, "CRITICAL": CRITICAL, "EMERGENCY": EMERGENCY
    }

    @staticmethod
    def get_level(priority_str: str) -> int:
        return NotificationPriority.STR_TO_LEVEL.get(priority_str.upper(), NotificationPriority.INFO)


class SimpleNotificationManager:
    """Manages sending notifications across various channels."""

    def __init__(self):
        self.notification_history = deque(maxlen=settings.NOTIFICATION_HISTORY_MAX_LENGTH)
        self.last_error_notification_time: Dict[str, datetime] = {} # Keyed by error type/message
        
        logger.info("Notification Manager initialized.")
        self._log_configured_channels()


    def _log_configured_channels(self):
        configured_channels = []
        if settings.SLACK_WEBHOOK_URL:
            configured_channels.append("Slack")
        if settings.SMTP_SERVER and settings.SENDER_EMAIL and settings.RECIPIENT_EMAIL:
            configured_channels.append("Email")
        if settings.DISCORD_WEBHOOK_URL:
            configured_channels.append("Discord")
        if settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN and settings.TWILIO_PHONE_NUMBER:
            configured_channels.append("SMS (Twilio)")
        
        if configured_channels:
            logger.info(f"Configured notification channels: {', '.join(configured_channels)}")
        else:
            logger.warning("No notification channels are fully configured.")

    async def notify(self, title: str, message: str, priority_str: str = "INFO"):
        """
        Sends a notification message with a given title and priority.
        Notifications are asynchronous and non-blocking.
        """
        priority_level = NotificationPriority.get_level(priority_str)
        min_priority_level = NotificationPriority.get_level(settings.MIN_NOTIFICATION_PRIORITY)

        if priority_level < min_priority_level:
            logger.debug(f"Notification '{title}' (Priority: {priority_str}) skipped due to low priority.")
            return

        timestamp = datetime.now()
        notification_entry = {
            "timestamp": timestamp.isoformat(),
            "title": title,
            "message": message,
            "priority": priority_str
        }
        self.notification_history.append(notification_entry)

        # Implement cooldown for ERROR notifications to prevent spamming
        # The .ERROR level will now be correctly interpreted
        if priority_level >= NotificationPriority.ERROR: 
            cooldown_key = f"{title}:{message[:50]}" # Use first 50 chars as key
            last_sent = self.last_error_notification_time.get(cooldown_key)
            if last_sent and (timestamp - last_sent).total_seconds() < settings.ERROR_NOTIFICATION_COOLDOWN:
                logger.debug(f"Notification '{title}' skipped due to cooldown.")
                return
            self.last_error_notification_time[cooldown_key] = timestamp

        logger.info(f"NOTIFICATION [{priority_str}] {title}: {message}")

        # Send notifications asynchronously to various channels
        tasks = []
        if settings.SLACK_WEBHOOK_URL:
            tasks.append(self._send_slack_notification(title, message, priority_str))
        if settings.SMTP_SERVER and settings.SENDER_EMAIL and settings.RECIPIENT_EMAIL:
            tasks.append(self._send_email_notification(title, message, priority_str))
        if settings.DISCORD_WEBHOOK_URL:
            tasks.append(self._send_discord_notification(title, message, priority_str))
        if settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN and settings.TWILIO_PHONE_NUMBER and settings.RECIPIENT_PHONE_NUMBER:
            tasks.append(self._send_sms_notification(title, message, priority_str))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True) # Run all tasks concurrently, log exceptions

    async def _send_slack_notification(self, title: str, message: str, priority: str):
        try:
            import httpx
            payload = {
                "username": settings.SLACK_USERNAME,
                "channel": settings.SLACK_CHANNEL,
                "text": f"*{title}* [{priority}]\n{message}"
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(settings.SLACK_WEBHOOK_URL, json=payload, timeout=5)
                response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

    async def _send_email_notification(self, title: str, message: str, priority: str):
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.utils import formataddr

            msg = MIMEText(f"Priority: {priority}\n\n{message}")
            msg['From'] = formataddr((f"{settings.APP_NAME} Bot", settings.SENDER_EMAIL))
            msg['To'] = settings.RECIPIENT_EMAIL
            msg['Subject'] = f"{settings.APP_NAME} Bot - {title}"

            async with asyncio.to_thread(smtplib.SMTP_SSL, settings.SMTP_SERVER, settings.SMTP_PORT) as server:
                await asyncio.to_thread(server.login, settings.SENDER_EMAIL, settings.SENDER_PASSWORD)
                await asyncio.to_thread(server.send_message, msg)
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    async def _send_discord_notification(self, title: str, message: str, priority: str):
        try:
            import httpx
            embed = {
                "title": f"{title} [{priority}]",
                "description": message,
                "color": self._get_discord_color(priority)
            }
            payload = {"embeds": [embed], "username": settings.APP_NAME}
            async with httpx.AsyncClient() as client:
                response = await client.post(settings.DISCORD_WEBHOOK_URL, json=payload, timeout=5)
                response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")

    async def _send_sms_notification(self, title: str, message: str, priority: str):
        try:
            from twilio.rest import Client
            client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
            body_message = f"{settings.APP_NAME} Bot [{priority}] - {title}: {message}"
            message_obj = await asyncio.to_thread(
                client.messages.create,
                to=settings.RECIPIENT_PHONE_NUMBER,
                from_=settings.TWILIO_PHONE_NUMBER,
                body=body_message
            )
            logger.debug(f"SMS sent: {message_obj.sid}")
        except ImportError:
            logger.error("Twilio library not installed. Cannot send SMS.")
        except Exception as e:
            logger.error(f"Failed to send SMS notification: {e}")

    def _get_discord_color(self, priority: str) -> int:
        colors = {
            "DEBUG": 0xAAAAAA, "INFO": 0x2ECC71, "WARNING": 0xF7D002, "ERROR": 0xFF6961, # New error color
            "HIGH": 0xE67E22, "CRITICAL": 0xE74C3C, "EMERGENCY": 0x9B59B6
        }
        return colors.get(priority.upper(), 0x3498DB) # Default to blue

    def get_notification_history(self) -> List[Dict[str, Any]]:
        """Returns a list of recent notifications from history."""
        return list(self.notification_history)

    def get_status_report(self) -> Dict[str, Any]:
        """Provides a status report for notification channels."""
        status = {
            "slack_configured": bool(settings.SLACK_WEBHOOK_URL),
            "email_configured": bool(settings.SMTP_SERVER and settings.SENDER_EMAIL and settings.RECIPIENT_EMAIL),
            "discord_configured": bool(settings.DISCORD_WEBHOOK_URL),
            "sms_configured": bool(settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN and settings.TWILIO_PHONE_NUMBER and settings.RECIPIENT_PHONE_NUMBER),
            "history_length": len(self.notification_history),
            "min_priority": settings.MIN_NOTIFICATION_PRIORITY,
            "cooldown_period_seconds": settings.ERROR_NOTIFICATION_COOLDOWN
        }
        return status