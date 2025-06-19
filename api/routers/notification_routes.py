# api/routers/notification_routes.py
import logging

from fastapi import APIRouter, Depends, Form, HTTPException, Request

from core.dependencies import get_notification_manager_dep # Import dependency provider

router = APIRouter(
    prefix="/api/notifications",
    tags=["Notifications"]
)

logger = logging.getLogger(__name__)

@router.get("/")
async def api_get_notifications(manager=Depends(get_notification_manager_dep)):
    """Retrieves the status and recent history of notifications."""
    status = manager.get_status()
    recent_history = manager.get_recent_notifications(20) # Get last 20

    return {
        "success": True,
        "enabled": status["enabled"],
        "channels": status["channels"],
        "history": recent_history,
        "total_sent": status["total_sent"]
    }

@router.post("/test")
async def api_test_notifications(manager=Depends(get_notification_manager_dep)):
    """Sends a test notification to all configured channels."""
    try:
        test_results = await manager.test_all_channels()
        await manager.send_notification(
            "Test Notification",
            "This is a system-generated test notification from your trading bot!",
            "medium"
        )
        return {
            "success": True,
            "message": "Test notification sent successfully.",
            "channel_results": test_results
        }
    except HTTPException: # Re-raise FastAPI's HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Failed to send test notifications: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to send test notification: {str(e)}")

@router.post("/send")
async def api_send_notification(
    request: Request, # Keep Request to distinguish JSON vs Form
    manager=Depends(get_notification_manager_dep)
):
    """Sends a custom notification with specified title, message, and priority."""
    try:
        content_type = request.headers.get("content-type", "").lower()

        if "application/json" in content_type:
            data = await request.json()
            title = data.get("title", "Custom Notification")
            message = data.get("message", "Test message")
            priority = data.get("priority", "medium")
        else:
            # Assume Form data if not JSON
            form = await request.form()
            title = form.get("title", "Custom Notification")
            message = form.get("message", "Test message")
            priority = form.get("priority", "medium")

        # Validate message content and length if necessary
        if not message:
            raise HTTPException(status_code=400, detail="Notification message cannot be empty.")

        await manager.send_notification(str(title), str(message), str(priority)) # Ensure str types
        return {
            "success": True,
            "message": "Notification sent successfully.",
            "notification": {"title": title, "message": message, "priority": priority}
        }
    except HTTPException: # Re-raise FastAPI's HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Failed to send custom notification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to send notification: {str(e)}")

@router.get("/history")
async def api_notification_history(manager=Depends(get_notification_manager_dep)):
    """Retrieves the full history of sent notifications."""
    try:
        history = manager.get_recent_notifications(manager.max_history) # Get all stored history
        return {
            "success": True,
            "history": history,
            "total_notifications": manager.get_status()["total_sent"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Notification history error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Notification history error: {str(e)}")