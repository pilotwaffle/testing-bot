# fix_main_py.py

import os
import sys
import re
from pathlib import Path

def apply_fixes_to_main_py(file_path="main.py"):
    """
    Applies known fixes to main.py, specifically:
    1. Replaces 'EnhancedTradingEngine' with 'IndustrialTradingEngine'.
    2. Ensures 'Optional' is imported from 'typing'.
    3. Ensures 'datetime' is imported from 'datetime'.
    4. Ensures 'load_dotenv' is imported and called at the top of the file.
    5. Corrects arguments passed to EnhancedChatManager's __init__ method.
    """
    
    print(f"🔍 Attempting to fix '{file_path}' for known errors...")

    if not Path(file_path).exists():
        print(f"❌ Error: File '{file_path}' not found. Please ensure this script is in the same directory as main.py.")
        return False

    lines_changed_count = 0
    new_lines = []

    # Flags to track if specific imports/calls are present or have been added
    found_typing_import = False
    optional_in_typing = False
    found_datetime_import = False
    found_load_dotenv_import = False
    found_load_dotenv_call_at_top = False
    
    # State for EnhancedChatManager block
    in_chat_manager_init_block = False
    chat_manager_init_indent = 0
    
    # Read the content of main.py
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Error reading '{file_path}': {e}")
        return False

    # First pass: Process lines to make replacements and identify import needs
    for i, line in enumerate(lines):
        original_line = line
        
        # --- Fix 1: Replace 'EnhancedTradingEngine' with 'IndustrialTradingEngine' ---
        if "EnhancedTradingEngine" in line and "IndustrialTradingEngine" not in line:
            line = line.replace("EnhancedTradingEngine", "IndustrialTradingEngine")
            if original_line != line:
                print(f"🔧 Replaced 'EnhancedTradingEngine' with 'IndustrialTradingEngine' on line {i+1}.")
                lines_changed_count += 1

        # --- Fix 2 & 3: Check for 'typing.Optional' and 'datetime' imports ---
        if "from typing import" in line:
            found_typing_import = True
            if "Optional" in line:
                optional_in_typing = True
            # If typing import exists but Optional is missing, add it
            elif "Optional" not in line and not re.search(r"from typing import \([^)]*\)", line): # Avoid complex multiline imports for simple add
                parts = line.strip().split("import ")
                if len(parts) > 1:
                    types = [t.strip() for t in parts[1].split(',')]
                    if 'Optional' not in types:
                        types.append('Optional')
                        line = f"{parts[0]}import {', '.join(sorted(types))}\n" # Sort for consistency
                        print(f"🔧 Added 'Optional' to 'from typing import' on line {i+1}.")
                        lines_changed_count += 1
                        optional_in_typing = True # Mark as fixed for this pass

        if "from datetime import datetime" in line:
            found_datetime_import = True

        # --- Fix 4: Check for 'dotenv' import and call ---
        if "from dotenv import load_dotenv" in line:
            found_load_dotenv_import = True
        if "load_dotenv()" in line and i < 10: # Assuming it should be near the top
             found_load_dotenv_call_at_top = True

        # --- Fix 5: Correct EnhancedChatManager initialization arguments ---
        if re.search(r"chat_manager\s*=\s*EnhancedChatManager\(", line):
            in_chat_manager_init_block = True
            chat_manager_init_indent = len(line) - len(line.lstrip())
            # Start a fresh line for the constructor for easier rebuilding
            # We'll reconstruct the arguments in the block
            current_line_content = line.strip()
            new_line_prefix = line[:chat_manager_init_indent]
            new_lines.append(f"{new_line_prefix}{current_line_content.split('(')[0].strip()} (\n")
            lines_changed_count += 1
            continue # Skip adding the current line as we'll reconstruct it

        if in_chat_manager_init_block:
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= chat_manager_init_indent and not line.strip() == '': # End of block
                in_chat_manager_init_block = False
                # If we're exiting, add the closing parenthesis for the constructor that was opened.
                # This assumes the closing ')' is on a separate line or at the end of the last argument line.
                # If the line contains only closing parenthesis, append it normally
                if line.strip() == ')':
                    new_lines.append(line) # Add the closing parenthesis
                elif line.strip().endswith(')'):
                    new_lines.append(line) # Add the closing parenthesis with potential comment

            # Process lines within the EnhancedChatManager constructor call
            if in_chat_manager_init_block:
                modified_arg = None
                
                if "trading_engine_instance=" in line:
                    modified_arg = line.replace("trading_engine_instance=", "trading_engine=")
                    modified_arg = modified_arg.replace(f"trading_engine,", "trading_engine=None, # Passed as None initially, set later\n") # Set to None and comment
                    if modified_arg != original_line:
                        print(f"🔧 Corrected 'trading_engine_instance' argument in EnhancedChatManager on line {i+1}.")
                        lines_changed_count += 1
                elif "ml_engine_instance=" in line:
                    modified_arg = line.replace("ml_engine_instance=", "ml_engine=")
                    if modified_arg != original_line:
                        print(f"🔧 Corrected 'ml_engine_instance' argument in EnhancedChatManager on line {i+1}.")
                        lines_changed_count += 1
                elif "data_fetcher_instance=" in line:
                    modified_arg = line.replace("data_fetcher_instance=", "data_fetcher=")
                    if modified_arg != original_line:
                        print(f"🔧 Corrected 'data_fetcher_instance' argument in EnhancedChatManager on line {i+1}.")
                        lines_changed_count += 1
                elif "notification_manager_instance=" in line:
                    modified_arg = line.replace("notification_manager_instance=", "notification_manager=")
                    if modified_arg != original_line:
                        print(f"🔧 Corrected 'notification_manager_instance' argument in EnhancedChatManager on line {i+1}.")
                        lines_changed_count += 1
                elif "google_ai_api_key=" in line:
                    print(f"🔧 Removed 'google_ai_api_key' argument from EnhancedChatManager on line {i+1}.")
                    lines_changed_count += 1
                    continue # Skip this line entirely
                
                if modified_arg is not None:
                    new_lines.append(modified_arg)
                else:
                    new_lines.append(line)
                continue # Processed line in block, move to next
        
        new_lines.append(line) # Lines not inside a special block

    # --- Second pass: Insert missing imports/calls if not found ---
    # This ensures that load_dotenv() is at the absolute top
    temp_final_output = []
    
    # Add load_dotenv at the very beginning if missing
    if not found_load_dotenv_import or not found_load_dotenv_call_at_top:
        if "from dotenv import load_dotenv" not in "".join(lines[:5]): # Check top 5 lines of original
            temp_final_output.append("from dotenv import load_dotenv\n")
            print("🔧 Added 'from dotenv import load_dotenv' at the top.")
            lines_changed_count += 1
        if "load_dotenv()" not in "".join(lines[:5]):
            temp_final_output.append("load_dotenv()\n")
            print("🔧 Added 'load_dotenv()' call at the top.")
            lines_changed_count += 1
    
    # Now append the rest of the content from the first pass
    for line in new_lines: # Use new_lines after first pass modifications
        # Avoid adding redundant dotenv lines if they were at the top but now moved
        if ("from dotenv import load_dotenv" in line and "from dotenv import load_dotenv\n" in temp_final_output) or \
           ("load_dotenv()" in line and "load_dotenv()\n" in temp_final_output):
            if temp_final_output.index(line) > 1 if line in temp_final_output else False: # check if already added at top
                continue
        temp_final_output.append(line)

    # Re-insert the datetime and Optional imports if still missing, now that dotenv is handled
    final_output_lines_with_header = []
    datetime_inserted = False
    
    for line in temp_final_output:
        final_output_lines_with_header.append(line)
        # Insert datetime import if not found and we are past initial imports/header
        if not found_datetime_import and not datetime_inserted and "import uvicorn" in line: # Insert after a common import point
            # Ensure it's not already there by some other means
            if "from datetime import datetime" not in "".join(final_output_lines_with_header):
                final_output_lines_with_header.insert(final_output_lines_with_header.index(line), "from datetime import datetime\n")
                print("🔧 Added 'from datetime import datetime'.")
                lines_changed_count += 1
                datetime_inserted = True
        
        # Ensure Optional is in typing import
        if not optional_in_typing and "from typing import" in line and "Optional" not in line:
            # Re-process the line to ensure Optional is there, if it wasn't added in the first pass
            parts = line.strip().split("import ")
            if len(parts) > 1:
                types = [t.strip() for t in parts[1].split(',')]
                if 'Optional' not in types:
                    types.append('Optional')
                    corrected_typing_line = f"{parts[0]}import {', '.join(sorted(types))}\n"
                    final_output_lines_with_header[final_output_lines_with_header.index(line)] = corrected_typing_line
                    print(f"🔧 Ensured 'Optional' in 'from typing import' on a later pass.")
                    lines_changed_count += 1
                    optional_in_typing = True # Mark as fixed

    # Write the modified content back to the file
    if lines_changed_count > 0:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(final_output_lines_with_header)
            print(f"\n✅ Successfully applied {lines_changed_count} corrections to '{file_path}'.")
            print("Please try running your bot again with: `python -m uvicorn main:app --host 0.0.0.0 --port 8000`")
            return True
        except Exception as e:
            print(f"❌ Error writing changes to '{file_path}': {e}")
            return False
    else:
        print(f"\n🎉 No specific corrections needed for '{file_path}' based on known patterns.")
        return True

if __name__ == "__main__":
    apply_fixes_to_main_py()