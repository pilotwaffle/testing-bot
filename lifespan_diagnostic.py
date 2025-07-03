# Filename: lifespan_diagnostic.py
# Location: Root directory of your G Trading Bot project (e.g., E:\Trade Chat Bot\G Trading Bot\)

import asyncio
import logging
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
import re
import subprocess
from textwrap import indent

# --- Configuration for the Diagnostic ---
MAIN_PY_PATH = Path("./main.py")
# The initial value for TIMEOUT_PER_STEP_SECONDS is a placeholder; it will be updated by user input.
# This needs to be available globally for inject_checkpoints and temp_file_content formatting.
TIMEOUT_PER_STEP_SECONDS_DISPLAY = 10 # Default for cli prompt
TIMEOUT_PER_STEP_SECONDS_ACTUAL = TIMEOUT_PER_STEP_SECONDS_DISPLAY

LOG_FILE = Path("./logs/diagnostic_output.log") # Log diagnostic script's own output

# Ensure the logs directory exists
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Setup logging for the diagnostic script itself
# Removed emoji from format string for better cross-platform/console compatibility.
# Explicitly set file encoding to UTF-8 for robustness.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding='utf-8')
    ]
)
logger = logging.getLogger("diagnostic")

# --- Helper Function to Extract Lifespan Body ---
def extract_lifespan_body(main_content: str) -> str:
    """
    Extracts the body of the lifespan function from the main.py content.
    Makes the regex for the function signature more robust and handles indentation for extraction.
    """
    # Normalize tabs to spaces, assuming 4 spaces per tab for consistency within the file
    main_content = main_content.replace('\t', '    ')

    # Robust pattern for `lifespan` function definition:
    # Handles variations in `app` parameter name and type hint (`app: FastAPI`, `app`, `app: Any`, etc.)
    lifespan_start_pattern = re.compile(
        r"(@asynccontextmanager\s*async\s+def\s+lifespan\s*\(\s*(?P<app_param_name>\w+)(?::\s*[\w\.]+)?\s*\):)",
        re.DOTALL # Allows '.' to match newlines, important for multi-line definitions if any.
    )
    
    match_start = lifespan_start_pattern.search(main_content)
    if not match_start:
        raise ValueError("Could not find @asynccontextmanager lifespan function in main.py. "
                         "Please ensure its definition matches 'async def lifespan(app: FastAPI):' or similar pattern.")

    # Find the start of the function definition line
    start_of_definition_line = main_content.rfind('\n', 0, match_start.start()) + 1
    # Get the definition line to determine its indentation
    definition_line = main_content[start_of_definition_line : match_start.end()]
    func_def_indent = len(definition_line) - len(definition_line.lstrip())
    
    # Expected indentation for the function body (usually 4 spaces more than definition)
    expected_body_indent = func_def_indent + 4

    # Extract lines from just after the definition to the end of the file
    lines_after_definition = main_content[match_start.end():].splitlines()
    
    lifespan_body_lines = []
    in_function_body = False

    for line in lines_after_definition:
        stripped_line = line.strip()
        current_indent = len(line) - len(stripped_line)

        # Logic to find the first line of the function's actual body
        if not in_function_body:
            if stripped_line and current_indent >= expected_body_indent:
                in_function_body = True
            else:
                continue # Skip leading empty lines or less-indented lines outside the body

        if in_function_body:
            # If we encounter a non-empty line with less indentation, the body has ended.
            # This captures the entire logically indented block of the lifespan function.
            if stripped_line and current_indent < expected_body_indent:
                break
            
            lifespan_body_lines.append(line)
            
    # Determine the minimum indentation of the extracted body lines to "normalize" them
    # This prepares the block to be re-indented correctly when placed into the temp script.
    min_indent = float('inf')
    for line in lifespan_body_lines:
        stripped = line.lstrip()
        if stripped: # Only consider non-empty lines for indentation calculation
            min_indent = min(min_indent, len(line) - len(stripped))
            
    if min_indent != float('inf') and min_indent > 0:
        lifespan_body_lines = [line[min_indent:] for line in lifespan_body_lines]
    
    return "\n".join(lifespan_body_lines)


# --- Function to Inject Checkpoints and Timeouts ---
def inject_checkpoints(lifespan_body: str) -> str:
    """
    Injects print statements and asyncio.wait_for around key initialization points.
    All f-string curly braces that are part of the *generated* code are escaped ({{ and }}).
    """
    lines = lifespan_body.splitlines()
    new_lines = []
    step_counter = 0

    # Patterns to look for initialization steps (can be expanded based on your main.py)
    # These should target lines that execute significant logic like constructors or async calls.
    init_patterns = [
        r"config_manager_instance\s*=", # Config init
        r"database_manager_instance\s*=", # DB init (constructor)
        r"await\s+database_manager_instance\.initialize\(\)", # DB init (async call)
        r"ml_engine_instance\s*=\s*(?:OctoBotMLEngine|.*?ml_engine)", # ML init (constructor/assignment for named engines)
        r"data_fetcher_instance\s*=\s*(?:CryptoDataFetcher|.*?data_fetcher)", # Data fetcher init
        r"notification_manager_instance\s*=", # Notification init
        r"risk_manager_instance\s*=\s*(?:RiskManager|.*?risk_management)", # Risk manager init
        r"trading_engine_instance\s*=\s*(?:IndustrialTradingEngine|.*?trading_engine)", # Trading engine init (constructor)
        r"await\s+trading_engine_instance\.start\(\)", # Trading engine start
        r"enhanced_chat_manager_instance\s*=\s*(?:EnhancedChatManager|.*?chat_manager)", # Chat manager init
        r"await\s+notification_manager_instance\.notify\(\s*\"CRITICAL\s+STARTUP\s+FAILURE\"" # Specific critical startup failure notification attempt
    ]

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        # Get leading whitespace for re-insertion
        current_indent_str = line[:len(line) - len(stripped_line)] if stripped_line else line
        
        # Check if this line is part of a multi-line comment/docstring, avoid injecting there
        if len(stripped_line) >= 3 and (stripped_line.startswith('"""') or stripped_line.startswith("'''")):
            new_lines.append(line)
            continue

        # Look for a pattern match on the current line
        match_found = False
        for pattern in init_patterns:
            if re.search(pattern, stripped_line):
                step_counter += 1
                new_lines.append(f"{current_indent_str}print(f'DIAGNOSTIC_START_STEP:{{step_counter}}:Attempting: {{stripped_line[:100]}}...')") # Note the escaped {{}}
                new_lines.append(f"{current_indent_str}_start_time = time.time()")
                
                # Check if the line contains an 'await' call that can be wrapped in await_for
                if stripped_line.startswith("await "):
                    actual_call = stripped_line[len("await "):]
                    # Indent the try/except block an additional level (e.g., 4 spaces more)
                    inner_indent = current_indent_str + ' ' * 4
                    try_block = (
# THIS IS A RAW STRING. No f-prefix. All { and } inside are literal unless doubled {{}}
f"""{current_indent_str}try:
{inner_indent}response = await asyncio.wait_for({actual_call}, timeout={{timeout_per_step_seconds_placeholder}})
{inner_indent}print(f'DIAGNOSTIC_END_STEP:{{step_counter}}:PASSED: {{time.time() - _start_time:.2f}}s')
{current_indent_str}except asyncio.TimeoutError:
{inner_indent}print(f'DIAGNOSTIC_END_STEP:{{step_counter}}:TIMEOUT: {{time.time() - _start_time:.2f}}s')
{inner_indent}raise asyncio.TimeoutError(f"Step {{step_counter}} timed out") # Re-raise with custom message
{current_indent_str}except Exception as e:
{inner_indent}print(f'DIAGNOSTIC_END_STEP:{{step_counter}}:FAILED: {{type(e).__name__}}: {{e}} (Line: {{i+1}})') # Add line num for context
{inner_indent}raise # Re-raise to propagate other errors"""
                    )
                    new_lines.append(try_block)
                else: # For assignments or simple non-async function calls (constructors)
                    # For these, we just add the original line.
                    # The subprocess timeout will catch if the call itself is blocking.
                    new_lines.append(line)
                    # Add a success print immediately after the original line
                    new_lines.append(f"{current_indent_str}print(f'DIAGNOSTIC_END_STEP:{{step_counter}}:PASSED: {{time.time() - _start_time:.2f}}s')") # Escaped {{}}
                
                match_found = True
                break # Only inject for the first matching pattern on this line

        if not match_found:
            new_lines.append(line)
    
    # Add a final checkpoint before `yield` to confirm all startup logic completed
    if any("yield" in line for line in lines): # Check if 'yield' is actually in the original extracted body
        # Find the line containing `yield` in the already partially processed `new_lines`
        yield_line_index = -1
        for i, line in enumerate(new_lines):
            # Ensure it's the actual 'yield' keyword and not just "yield" in a comment/string
            if "yield" in line.strip() and not line.strip().startswith("#"):
                yield_line_index = i
                break
        
        if yield_line_index != -1: # If 'yield' was found and processed
            step_counter += 1
            final_check_id = step_counter
            # Get the indent of the `yield` line to apply to the added block
            yield_line_indent = new_lines[yield_line_index][:len(new_lines[yield_line_index]) - len(new_lines[yield_line_index].lstrip())]

            yield_check_block = (
# THIS IS A RAW STRING. No f-prefix. All { and } inside are literal unless doubled {{}}
f"""{yield_line_indent}print('DIAGNOSTIC_START_STEP:{{final_check_id}}:Pre-yield check: All startup steps completed.')
{yield_line_indent}_start_time = time.time()
{yield_line_indent}try:
{yield_line_indent}    await asyncio.wait_for(asyncio.sleep(0.01), timeout={{timeout_per_step_seconds_placeholder}}) # Short sleep to ensure event loop processes
{yield_line_indent}    print(f'DIAGNOSTIC_END_STEP:{{final_check_id}}:PASSED: Pre-yield check finished in {{time.time() - _start_time:.2f}}s')
{yield_line_indent}except asyncio.TimeoutError:
{yield_line_indent}    print(f'DIAGNOSTIC_END_STEP:{{final_check_id}}:TIMEOUT: {{time.time() - _start_time:.2f}}s')
{yield_line_indent}    raise asyncio.TimeoutError(f"Step {{final_check_id}} (pre-yield) timed out")
{yield_line_indent}except Exception as e:
{yield_line_indent}    print(f'DIAGNOSTIC_END_STEP:{{final_check_id}}:FAILED: Pre-yield check error: {{type(e).__name__}}: {{e}}')
{yield_line_indent}    raise"""
            )
            new_lines.insert(yield_line_index, yield_check_block)

    return "\n".join(new_lines)


# --- Main Diagnostic Function ---
def run_diagnostic():
    global TIMEOUT_PER_STEP_SECONDS_ACTUAL # Declare as global to modify it

    # Get timeout from user input, or use default
    try:
        user_timeout = input(f"Enter timeout per initialization step in seconds (default {TIMEOUT_PER_STEP_SECONDS_DISPLAY}): ")
        if user_timeout.strip():
            TIMEOUT_PER_STEP_SECONDS_ACTUAL = float(user_timeout)
            if TIMEOUT_PER_STEP_SECONDS_ACTUAL <= 0:
                raise ValueError("Timeout must be positive.")
    except ValueError as e:
        logger.warning(f"Invalid timeout value: {e}. Using default {TIMEOUT_PER_STEP_SECONDS_DISPLAY} seconds.")

    logger.info("Starting Enhanced Trading Bot Lifespan Diagnostic")
    logger.info(f"Target main.py: {MAIN_PY_PATH.resolve()}")
    logger.info(f"Timeout per initialization step: {TIMEOUT_PER_STEP_SECONDS_ACTUAL} seconds")

    if not MAIN_PY_PATH.exists():
        logger.error(f"Error: main.py not found at {MAIN_PY_PATH.resolve()}")
        sys.exit(1)

    try:
        main_content = MAIN_PY_PATH.read_text(encoding='utf-8')
        lifespan_body = extract_lifespan_body(main_content)
        if not lifespan_body.strip():
            logger.error("Could not extract non-empty lifespan function body. This might indicate main.py structure issues.")
            sys.exit(1)
        
        # Inject checkpoints into the extracted lifespan body
        # Pass the desired timeout value as a placeholder string to be formatted later
        # The instrumented_lifespan_body will contain "timeout={timeout_per_step_seconds_placeholder}"
        instrumented_lifespan_body_with_placeholder = inject_checkpoints(lifespan_body) 
        
        # Estimate theoretical max steps for subprocess timeout
        # Count explicit patterns + 1 for the pre-yield check + 3 for shutdown steps executed within lifespan
        approx_total_patterns = instrumented_lifespan_body_with_placeholder.count("DIAGNOSTIC_START_STEP:")
        global_subprocess_timeout = TIMEOUT_PER_STEP_SECONDS_ACTUAL * (approx_total_patterns + 5) # +5 for buffer and shutdown
        logger.info(f"Anticipated maximum total subprocess timeout: {round(global_subprocess_timeout, 2)} seconds (estimated diagnostic steps: {approx_total_patterns})")

        # --- Construct the temporary diagnostic file content ---
        # This is built as a list of strings and then joined.
        # Use .format() at the end to inject variables from *this* script.
        temp_file_lines = []

        # Part 1: Standard boilerplate imports and mock classes for FastAPI and basic types
        # NO f-string for append. All { and } meant for the GENERATED file are escaped.
        temp_file_lines.append("""
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# --- Mock FastAPI components to prevent actual server startup ---
class MockRequest:
    pass

class MockHTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP Error {{status_code}}: {{detail}}")

class MockFastAPI:
    def __init__(self, lifespan=None, title='MockApp', version='1.0', description='Mock'):
        self.lifespan_context = lifespan
        self.routes = []
    def mount(self, *args, **kwargs): pass
    def include_router(self, *args, **kwargs): pass
    def get(self, *args, **kwargs): return lambda func: func
    def post(self, *args, **kwargs): return lambda func: func
    def delete(self, *args, **kwargs): return lambda func: func
    def websocket(self, *args, **kwargs): return lambda func: func

class MockJinja2Templates:
    def __init__(self, directory): pass
    def TemplateResponse(self, template_name, context):
        print(f"DIAGNOSTIC_INFO: Jinja2Templates trying to render {{template_name}}")
        return f"Mocked HTML for {{template_name}}"

class MockUvicorn:
    @staticmethod
    def run(app, **kwargs):
        print("DIAGNOSTIC_INFO: Uvicorn.run called, mocking to prevent server start.")

# --- Re-import core components from the actual project structure ---
# Add the bot's root directory to Python's sys.path to resolve imports like 'core.config'.
# Path(__file__).parent is the directory containing this temporary script (which is the bot's root).
sys.path.insert(0, str(Path(__file__).parent)) 

# Mock settings. Just enough to prevent crashes.
class MockSettings:
    DEBUG = True
    APP_NAME = "DiagnosticBot"
    APP_USER_ID = "diagnostic_user"
    GOOGLE_AI_ENABLED = False 

settings = MockSettings()

# --- Component Mocks and Imports (Graceful Handling) ---
""")

        # Part 2: Mock implementations of your bot's core components
        # NO f-string for append. All { and } meant for the GENERATED file are escaped.
        temp_file_lines.append("""
try:
    from core.config import ConfigManager
except ImportError:
    print("DIAGNOSTIC_WARNING: Could not import ConfigManager. Mocking it.")
    class ConfigManager:
        def load_config(self, path): 
            print(f"DIAGNOSTIC_INFO: Mocking ConfigManager.load_config({{path}})")
            return {{"trading": {{"dry_run": True, "max_open_trades": 1}}, "exchange": {{"enabled": False}}}}

try:
    from core.notification_manager import SimpleNotificationManager
except ImportError:
    print("DIAGNOSTIC_WARNING: Could not import SimpleNotificationManager. Mocking it.")
    class SimpleNotificationManager:
        async def notify(self, *args, **kwargs):
            print(f"DIAGNOSTIC_INFO: Mocking SimpleNotificationManager.notify with args: {{args}}, kwargs: {{kwargs}}")

try:
    # IMPORTANT: Adjust this import path if IndustrialTradingEngine is not in core/trading_engine.py
    from core.trading_engine import IndustrialTradingEngine 
except ImportError:
    print("DIAGNOSTIC_WARNING: Could not import IndustrialTradingEngine. Mocking it.")
    class IndustrialTradingEngine:
        def __init__(self, notification_manager, config):
            print("DIAGNOSTIC_INFO: Mocking IndustrialTradingEngine.__init__")
            self.running = False
            self.current_market_data = {{}}
            self.balances = {{}}
            self.positions = {{}}

        async def start(self): 
            print("DIAGNOSTIC_INFO: Mocking IndustrialTradingEngine.start()")
            self.running = True
            await asyncio.sleep(0.01) # Simulate async work
        async def stop(self): 
            print("DIAGNOSTIC_INFO: Mocking IndustrialTradingEngine.stop()")
            self.running = False
            await asyncio.sleep(0.01) # Simulate async work
        def get_status(self): return {{"status": "mocked","running":self.running}}
        def get_enhanced_status(self): return {{"status": "mocked_enhanced","running":self.running}}
        def get_performance_metrics(self): return {{"performance": "mocked"}}
        def list_available_strategies(self): return []
        def list_active_strategies(self): return []
        def add_strategy(self, *args): return True
        def remove_strategy(self, *args): return True

try:
    from core.database import DatabaseManager
    DATABASE_AVAILABLE = True
except ImportError:
    print("DIAGNOSTIC_WARNING: Could not import DatabaseManager. Mocking it.")
    DATABASE_AVAILABLE = False
    class DatabaseManager:
        def __init__(self, db_config={{}}): print("DIAGNOSTIC_INFO: Mocking DatabaseManager.__init__")
        async def initialize(self):
            print("DIAGNOSTIC_INFO: Mocking DatabaseManager.initialize()")
            await asyncio.sleep(0.01) # Simulate async work
        async def close(self):
            print("DIAGNOSTIC_INFO: Mocking DatabaseManager.close()")
            await asyncio.sleep(0.01) # Simulate async work

try:
    from core.risk_manager import RiskManager
    RISK_MANAGEMENT_AVAILABLE = True
except ImportError:
    print("DIAGNOSTIC_WARNING: Could not import RiskManager. Mocking it.")
    RISK_MANAGEMENT_AVAILABLE = False
    class RiskManager:
        def __init__(self, config={{}}): print("DIAGNOSTIC_INFO: Mocking RiskManager.__init__")

try:
    # IMPORTANT: Adjust this import path if OctoBotMLEngine is not in ml/ml_engine.py
    from ml.ml_engine import OctoBotMLEngine
    ML_ENGINE_AVAILABLE = True
except ImportError:
    print("DIAGNOSTIC_WARNING: Could not import OctoBotMLEngine. Mocking it.")
    ML_ENGINE_AVAILABLE = False
    class OctoBotMLEngine:
        def __init__(self): print("DIAGNOSTIC_INFO: Mocking OctoBotMLEngine.__init__")

try:
    # IMPORTANT: Adjust this import path if CryptoDataFetcher is not in core/data_fetcher.py
    from core.data_fetcher import CryptoDataFetcher
    DATA_FETCHER_AVAILABLE = True
except ImportError:
    print("DIAGNOSTIC_WARNING: Could not import CryptoDataFetcher. Mocking it.")
    DATA_FETCHER_AVAILABLE = False
    class CryptoDataFetcher:
        def __init__(self): print("DIAGNOSTIC_INFO: Mocking CryptoDataFetcher.__init__") 

try:
    # IMPORTANT: Adjust this import path if EnhancedChatManager is not in ai/chat_manager.py
    from ai.chat_manager import EnhancedChatManager
    ENHANCED_CHAT_AVAILABLE = True
except ImportError:
    print("DIAGNOSTIC_WARNING: Could not import EnhancedChatManager. Mocking it.")
    ENHANCED_CHAT_AVAILABLE = False
    class EnhancedChatManager:
        def __init__(self, **kwargs):
            print("DIAGNOSTIC_INFO: Mocking EnhancedChatManager.__init__")
            self.memory = type('MemoryMock', (object,), {{"short_term": [], "topic_threads": [], "session_start": datetime.now()}})()
            self.response_times = [1] # Avoid division by zero in status checks
        def process_message(self, *args, **kwargs): 
            print("DIAGNOSTIC_INFO: Mocking EnhancedChatManager.process_message")
            return {{"response": "Mocked AI response", "message_type": "text", "response_time": 0.1}}

# --- Pydantic Mocks (if not directly importable or preferred for simplicity) ---
try:
    from pydantic import BaseModel
except ImportError:
    print("DIAGNOSTIC_WARNING: Pydantic not found, using basic class for BaseModel.")
    class BaseModel:
        def __init__(self, **kwargs): 
            for k,v in kwargs.items(): setattr(self, k, v)
        @classmethod
        def __get_validators__(cls): yield (lambda x: x) # Corrected to yield a lambda
        @classmethod
        def __modify_schema__(cls, field_schema): pass
        
# Define Pydantic models needed for the lifespan function 
class StrategyConfig(BaseModel):
    id: str = "mock"
    type: str = "mock"
    config: Dict[str, Any] = {{}}
    symbols: List[str] = []
    enabled: bool = True

class TradeRequest(BaseModel): pass
class EnhancedChatMessage(BaseModel): pass
class NotificationRequest(BaseModel): pass

# Global instances (init to None, will be set by lifespan)
notification_manager_instance = None
trading_engine_instance = None
enhanced_chat_manager_instance = None
config_manager_instance = None
database_manager_instance = None
risk_manager_instance = None
ml_engine_instance = None
data_fetcher_instance = None

# Configure logging for the temporary script to direct all output to stdout
logger_out = logging.getLogger("crypto-bot") 
logger_out.setLevel(logging.DEBUG) 
logger_out.handlers = [] 
logger_out.addHandler(logging.StreamHandler(sys.stdout)) 

print("DIAGNOSTIC_INFO: Temporary diagnostic environment initialized.")

@asynccontextmanager
async def lifespan_mock(app):
    global notification_manager_instance
    global trading_engine_instance
    global enhanced_chat_manager_instance
    global config_manager_instance
    global database_manager_instance
    global risk_manager_instance
    global ml_engine_instance
    global data_fetcher_instance

    # Instrumented body of your main.py's lifespan function will be inserted here
""")
        # Part 3: The instrumented lifespan body from main.py's content
        # This part already contains correct f-strings and escaped {{}} for the inner generated code.
        # It also contains the placeholder `{timeout_per_step_seconds_placeholder}`.
        # This is indented to fit correctly within the lifespan_mock function block.
        temp_file_lines.append(indent(instrumented_lifespan_body_with_placeholder, '    '))

        # Part 4: Shutdown sequence and final test runners
        # NO f-string for append. Use .format() placeholders for values from the outer script.
        temp_file_lines.append("""
    # The `yield` is crucial for the lifespan context manager.
    # It signifies the entry into the application's runtime phase.
    yield 
    
    # --- Shutdown sequence (copied and instrumented) ---
    logger_out.info("DIAGNOSTIC_INFO: Entering lifespan shutdown sequence...")
    print("DIAGNOSTIC_START_STEP:Shutdown:Attempting graceful shutdown...")
    _start_time = time.time()
    try:
        if trading_engine_instance and trading_engine_instance.running:
            await asyncio.wait_for(trading_engine_instance.stop(), timeout={outer_script_timeout})
            print(f'DIAGNOSTIC_END_STEP:Shutdown:PASSED: Trading engine stopped in {{time.time() - _start_time:.2f}}s')
        
        if database_manager_instance:
            await asyncio.wait_for(database_manager_instance.close(), timeout={outer_script_timeout})
            print(f'DIAGNOSTIC_END_STEP:Shutdown:PASSED: Database closed in {{time.time() - _start_time:.2f}}s')
        
        logger_out.info("DIAGNOSTIC_INFO: Shutdown sequence completed.")
    except asyncio.TimeoutError:
        print(f'DIAGNOSTIC_END_STEP:Shutdown:TIMEOUT: A shutdown step timed out after {{time.time() - _start_time:.2f}}s')
    except Exception as e:
        print(f'DIAGNOSTIC_END_STEP:Shutdown:FAILED: {{type(e).__name__}}: {{e}}')
        logger_out.exception(f"DIAGNOSTIC_INFO: Exception during shutdown:") 
    print("DIAGNOSTIC_INFO: Temporary diagnostic shutdown complete.")

async def run_test():
    """
    Sets up a mock FastAPI app and runs its lifespan context.
    """
    mock_app = MockFastAPI(lifespan=lifespan_mock) # Corrected typo: lifespan_mock
    async with mock_app.lifespan_context(mock_app):
        print("DIAGNOSTIC_INFO: lifespan context entered. All startup steps should have completed.")
        await asyncio.sleep(0.5) 
    print("DIAGNOSTIC_INFO: lifespan context exited. Shutdown complete.")

if __name__ == "__main__":
    asyncio.run(run_test())
""")
        # Concatenate all lines to form the final content
        temp_file_content_raw = "\n".join(temp_file_lines)
        
        # Now, format the placeholders with the actual values from this script.
        final_temp_file_content = temp_file_content_raw.format(
            timeout_per_step_seconds_placeholder=TIMEOUT_PER_STEP_SECONDS_ACTUAL,
            outer_script_timeout=TIMEOUT_PER_STEP_SECONDS_ACTUAL # Use the same timeout for shutdown steps
        )

        # Use a temporary file to write the diagnostic script
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".py", encoding='utf-8') as temp_f:
            temp_f.write(final_temp_file_content) # Write the fully formatted content
            temp_script_path = Path(temp_f.name)
        
        logger.info(f"Temporary diagnostic script created at: {temp_script_path}")
        logger.info("Executing diagnostic script in a subprocess. Please wait. All output will be logged here.")

        start_total_time = time.time()
        process = subprocess.Popen(
            [sys.executable, str(temp_script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=MAIN_PY_PATH.parent, # Set CWD to the bot's root directory for imports
            env={**os.environ, 'PYTHONUNBUFFERED': '1'} # Ensure unbuffered output for real-time logging
        )

        stdout_buffer = []
        stderr_buffer = []
        timeout_occurred = False
        
        # Read output line by line with a global timeout for the subprocess
        try:
            while True:
                # Poll stdout and stderr to avoid deadlocks
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()

                if not stdout_line and not stderr_line and process.poll() is not None:
                    break # Process finished

                if stdout_line:
                    stdout_buffer.append(stdout_line.strip())
                    logger.info(f"Subprocess: {stdout_line.strip()}")
                
                if stderr_line:
                    stderr_buffer.append(stderr_line.strip())
                    logger.error(f"Subprocess ERR: {stderr_line.strip()}")

                # Check for global timeout
                if time.time() - start_total_time > global_subprocess_timeout:
                    process.terminate() # Request graceful termination
                    logger.critical(f"Global diagnostic timeout after {time.time() - start_total_time:.2f} seconds. Subprocess terminated.")
                    timeout_occurred = True
                    break # Break the loop, process will be killed in finally if needed

        except Exception as e:
            logger.critical(f"Error reading subprocess output: {e}", exc_info=True)
            process.kill() # Ensure subprocess is terminated on parent error
            raise
        finally:
            # Attempt to read any remaining output after termination/kill
            final_stdout, final_stderr = process.communicate(timeout=5) # Collect remaining output
            if final_stdout:
                stdout_buffer.extend(final_stdout.strip().splitlines())
                for line in final_stdout.strip().splitlines(): logger.info(f"Subprocess: {line}")
            if final_stderr:
                stderr_buffer.extend(final_stderr.strip().splitlines())
                for line in final_stderr.strip().splitlines(): logger.error(f"Subprocess ERR: {line}")

            # Ensure the process is fully terminated before exiting parent
            if process.poll() is None:
                process.kill() # Last resort to kill if still running

            # Clean up the temporary file regardless of success or failure
            if temp_script_path.exists():
                os.remove(temp_script_path)
                logger.info(f"Cleaned up temporary diagnostic script: {temp_script_path}")
            

        return_code = process.wait() # Get final return code of subprocess

        logger.info("\n--- Diagnostic Results ---")
        if timeout_occurred:
            logger.critical("DIAGNOSTIC: The diagnostic process timed out. A component likely hung without reporting.")
            logger.critical("Review the logs above. The last 'DIAGNOSTIC_START_STEP' message before the timeout indicates the hanging point.")
            logger.critical("If you see 'DIAGNOSTIC_END_STEP:X:TIMEOUT', that particular step caused the timeout.")
        elif return_code != 0:
            logger.critical(f"DIAGNOSTIC: The diagnostic script exited with an error (Code: {return_code}). See logs for details.")
            last_failed_step_match = None
            for line in reversed(stdout_buffer): # Search from end for the last failure
                if "DIAGNOSTIC_END_STEP" in line and ("FAILED" in line or "TIMEOUT" in line):
                    last_failed_step_match = line
                    break
            
            if last_failed_step_match:
                logger.critical(f"DIAGNOSTIC: Last recorded issue: {last_failed_step_match}")
            else:
                logger.critical("DIAGNOSTIC: No specific step failure message found in output before subprocess crash.")
                logger.critical("This could mean the crash occurred immediately after starting or before instrumented steps, or might be within a mocked component.")
        else:
            logger.info("DIAGNOSTIC: All instrumented steps completed successfully. No hang detected during initialization.")
            logger.info("This suggests the hanging issue might be outside the lifespan function's instrumented blocks,")
            logger.info("or that the problem is too nuanced for this level of instrumentation (e.g., a very long blocking call that isn't `await`ed).")
            logger.info("Look for other 'DIAGNOSTIC_INFO' or 'DIAGNOSTIC_WARNING' messages in the logs for clues.")
        
        logger.info(f"Total diagnostic run time: {time.time() - start_total_time:.2f} seconds")
        logger.info("------------------------")

    except Exception as e:
        logger.critical(f"An unexpected error occurred during diagnostic setup or execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_diagnostic()