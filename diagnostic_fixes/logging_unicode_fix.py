"""
File: logging_unicode_fix.py
Location: E:\Trade Chat Bot\G Trading Bot\diagnostic_fixes\logging_unicode_fix.py
Description: Unicode logging configuration fix for Windows console
"""

# Add this to your main.py or logging configuration:

import sys
import codecs
import logging

# Fix Windows console Unicode support
if sys.platform == "win32":
    # Reconfigure stdout/stderr for UTF-8
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    # Set environment variable
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configure logging with UTF-8 support
def setup_utf8_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('trading_bot.log', encoding='utf-8')
        ]
    )
