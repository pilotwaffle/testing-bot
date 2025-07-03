#!/usr/bin/env python3
"""
Enhanced Trading Engine Specific Diagnostic
Deep dive into the enhanced_trading_engine.py hanging issue
"""

import sys
import time
import ast
import re
from pathlib import Path
import traceback
import concurrent.futures
import concurrent.futures

class EnhancedEngineDiagnostic:
    def __init__(self, timeout_seconds=5):
        self.timeout_seconds = timeout_seconds
        self.engine_file = None
        self.find_engine_file()
    
    def find_engine_file(self):
        """Find the enhanced_trading_engine.py file"""
        possible_paths = [
            'enhanced_trading_engine.py',
            'core/enhanced_trading_engine.py',
            'engine/enhanced_trading_engine.py',
            'src/enhanced_trading_engine.py'
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                self.engine_file = path
                print(f"✅ Found enhanced_trading_engine.py at: {path}")
                return
        
        print("❌ Could not find enhanced_trading_engine.py")
        print("Please specify the correct path:")
        user_path = input("Enter path to enhanced_trading_engine.py: ")
        if Path(user_path).exists():
            self.engine_file = user_path
        else:
            print("❌ File not found. Exiting.")
            sys.exit(1)
    
    def safe_execute(self, operation_name, func, *args, **kwargs):
        """Execute a function with timeout protection (Windows compatible)"""
        print(f"\n🔍 Testing: {operation_name}")
        
        start_time = time.time()
        
        # Use ThreadPoolExecutor for Windows-compatible timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            
            try:
                result = future.result(timeout=self.timeout_seconds)
                elapsed = time.time() - start_time
                print(f"✅ PASSED ({elapsed:.2f}s): {operation_name}")
                return result
            except concurrent.futures.TimeoutError:
                elapsed = time.time() - start_time
                print(f"⏰ TIMEOUT ({elapsed:.2f}s): {operation_name} - THIS IS LIKELY THE HANGING POINT")
                future.cancel()
                return None
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"❌ FAILED ({elapsed:.2f}s): {operation_name} - {str(e)}")
                return None
    
    def analyze_file_structure(self):
        """Analyze the structure of enhanced_trading_engine.py"""
        print("\n📁 Analyzing File Structure...")
        
        try:
            with open(self.engine_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"✅ File size: {len(content)} characters")
            
            # Count lines
            lines = content.split('\n')
            print(f"✅ Total lines: {len(lines)}")
            
            # Find imports
            import_lines = [line.strip() for line in lines if line.strip().startswith(('import ', 'from '))]
            print(f"✅ Import statements: {len(import_lines)}")
            
            # Find classes
            class_lines = [line.strip() for line in lines if line.strip().startswith('class ')]
            print(f"✅ Classes found: {len(class_lines)}")
            for class_line in class_lines:
                print(f"   - {class_line}")
            
            # Find functions
            function_lines = [line.strip() for line in lines if line.strip().startswith('def ') and not line.strip().startswith('def __')]
            print(f"✅ Functions found: {len(function_lines)}")
            
            return content
            
        except Exception as e:
            print(f"❌ File analysis failed: {e}")
            return None
    
    def test_imports_individually(self):
        """Test each import statement individually"""
        print("\n📦 Testing Individual Imports...")
        
        try:
            with open(self.engine_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            import_lines = []
            
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(('import ', 'from ')) and not stripped.startswith('#'):
                    import_lines.append(stripped)
            
            print(f"Found {len(import_lines)} import statements to test:")
            
            for i, import_line in enumerate(import_lines, 1):
                self.safe_execute(
                    f"Import {i}/{len(import_lines)}: {import_line[:50]}...",
                    self.test_single_import,
                    import_line
                )
        
        except Exception as e:
            print(f"❌ Import testing failed: {e}")
    
    def test_single_import(self, import_statement):
        """Test a single import statement"""
        try:
            # Clean the import statement
            clean_import = import_statement.strip()
            if clean_import.endswith(','):
                clean_import = clean_import[:-1]
            
            # Execute the import
            exec(clean_import, {})
            return True
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            return False
    
    def test_class_definitions(self):
        """Test class definitions without instantiation"""
        print("\n🏗️ Testing Class Definitions...")
        
        try:
            with open(self.engine_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST to find classes
            try:
                tree = ast.parse(content)
                classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                
                print(f"Found {len(classes)} class definitions:")
                for cls in classes:
                    print(f"   - {cls.name} (line {cls.lineno})")
                    
                    # Test loading just the class definition
                    self.safe_execute(
                        f"Class Definition: {cls.name}",
                        self.test_class_syntax,
                        content,
                        cls.name,
                        cls.lineno
                    )
                    
            except SyntaxError as e:
                print(f"❌ Syntax error in file: {e}")
                
        except Exception as e:
            print(f"❌ Class definition testing failed: {e}")
    
    def test_class_syntax(self, content, class_name, line_no):
        """Test if a class definition has valid syntax"""
        lines = content.split('\n')
        
        # Find the class definition
        class_start = line_no - 1
        class_lines = []
        indent_level = None
        
        for i in range(class_start, len(lines)):
            line = lines[i]
            
            if i == class_start:
                class_lines.append(line)
                # Determine base indentation
                indent_level = len(line) - len(line.lstrip())
                continue
            
            if line.strip() == '':
                class_lines.append(line)
                continue
            
            current_indent = len(line) - len(line.lstrip())
            
            # If we hit a line with same or less indentation (and it's not empty), class is done
            if current_indent <= indent_level and line.strip():
                break
                
            class_lines.append(line)
        
        class_code = '\n'.join(class_lines)
        
        try:
            # Just parse, don't execute
            ast.parse(class_code)
            return True
        except SyntaxError as e:
            print(f"     ❌ Syntax error in {class_name}: {e}")
            return False
    
    def test_module_compilation(self):
        """Test if the entire module can be compiled"""
        print("\n🔧 Testing Module Compilation...")
        
        def compile_module():
            with open(self.engine_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            compile(content, self.engine_file, 'exec')
            return True
        
        return self.safe_execute("Module Compilation", compile_module)
    
    def test_module_import(self):
        """Test importing the module"""
        print("\n📥 Testing Module Import...")
        
        def import_module():
            import importlib.util
            
            # Get module name from file path
            module_name = Path(self.engine_file).stem
            
            spec = importlib.util.spec_from_file_location(module_name, self.engine_file)
            module = importlib.util.module_from_spec(spec)
            
            # This is where it might hang
            spec.loader.exec_module(module)
            return module
        
        return self.safe_execute("Module Import", import_module)
    
    def find_problematic_sections(self):
        """Identify potentially problematic code sections"""
        print("\n🔍 Scanning for Problematic Code Patterns...")
        
        try:
            with open(self.engine_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Patterns that commonly cause hangs
            problematic_patterns = [
                (r'while\s+True:', 'Infinite while loop'),
                (r'for.*in.*range\(\s*\d{4,}', 'Large range loop'),
                (r'time\.sleep\(\s*\d+', 'Long sleep call'),
                (r'requests\.get\(.*timeout=None', 'Request without timeout'),
                (r'\.join\(\)', 'Thread join without timeout'),
                (r'input\(', 'Blocking input call'),
                (r'tf\.|tensorflow\.', 'TensorFlow import/usage'),
                (r'keras\.', 'Keras import/usage'),
                (r'torch\.', 'PyTorch import/usage'),
                (r'ccxt\..*\(\)', 'CCXT exchange calls'),
                (r'yfinance\.download', 'Yahoo Finance download'),
                (r'\.connect\(\)', 'Database/network connection'),
                (r'subprocess\.', 'Subprocess call'),
            ]
            
            issues_found = []
            
            for line_no, line in enumerate(lines, 1):
                for pattern, description in problematic_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues_found.append((line_no, line.strip(), description))
                        print(f"⚠️  Line {line_no}: {description}")
                        print(f"     {line.strip()}")
            
            if not issues_found:
                print("✅ No obvious problematic patterns found")
            else:
                print(f"\n📊 Found {len(issues_found)} potentially problematic lines")
            
            return issues_found
            
        except Exception as e:
            print(f"❌ Pattern scanning failed: {e}")
            return []
    
    def test_step_by_step_execution(self):
        """Execute the module line by line to find hanging point"""
        print("\n🐛 Step-by-Step Execution Test...")
        
        try:
            with open(self.engine_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            print(f"Testing execution of {len(lines)} lines...")
            
            # Create a test namespace
            test_namespace = {'__name__': '__main__'}
            
            accumulated_code = ""
            
            for line_no, line in enumerate(lines, 1):
                if line.strip() and not line.strip().startswith('#'):
                    accumulated_code += line + '\n'
                    
                    # Test every 10 lines or on class/function definitions
                    if (line_no % 10 == 0 or 
                        line.strip().startswith(('class ', 'def ', 'import ', 'from '))):
                        
                        result = self.safe_execute(
                            f"Lines 1-{line_no}",
                            self.execute_code_block,
                            accumulated_code,
                            test_namespace.copy()
                        )
                        
                        if result is None:  # Timeout or error
                            print(f"🚨 HANGING DETECTED at or before line {line_no}:")
                            print(f"     {line.strip()}")
                            
                            # Show surrounding lines for context
                            start = max(0, line_no - 5)
                            end = min(len(lines), line_no + 5)
                            
                            print(f"\n📝 Context (lines {start+1}-{end}):")
                            for i in range(start, end):
                                marker = ">>> " if i == line_no - 1 else "    "
                                print(f"{marker}{i+1:4d}: {lines[i]}")
                            
                            return line_no
                            
        except Exception as e:
            print(f"❌ Step-by-step execution failed: {e}")
            return None
    
    def execute_code_block(self, code, namespace):
        """Execute a block of code in a given namespace"""
        try:
            exec(code, namespace)
            return True
        except Exception as e:
            # Some errors are expected during partial execution
            return False
    
    def run_comprehensive_diagnostic(self):
        """Run all diagnostic tests"""
        print("🚀 Enhanced Trading Engine Diagnostic")
        print("=" * 50)
        print(f"Target file: {self.engine_file}")
        print(f"Timeout per test: {self.timeout_seconds} seconds")
        print("=" * 50)
        
        # Test 1: File structure analysis
        content = self.analyze_file_structure()
        if not content:
            return
        
        # Test 2: Pattern scanning
        self.find_problematic_sections()
        
        # Test 3: Module compilation
        if self.test_module_compilation():
            print("✅ Module compiles successfully")
        else:
            print("❌ Module compilation failed - fix syntax errors first")
            return
        
        # Test 4: Individual imports
        self.test_imports_individually()
        
        # Test 5: Class definitions
        self.test_class_definitions()
        
        # Test 6: Full module import
        if self.test_module_import():
            print("✅ Module imports successfully - no hanging detected!")
        else:
            print("🚨 Module import failed/hung - running step-by-step analysis...")
            
            # Test 7: Step-by-step execution
            hanging_line = self.test_step_by_step_execution()
            if hanging_line:
                print(f"\n🎯 CONCLUSION: Hanging occurs at or before line {hanging_line}")
            else:
                print("\n🤔 Unable to pinpoint exact hanging location")
        
        print("\n🏁 Enhanced Engine Diagnostic Complete!")

def main():
    print("🔧 Enhanced Trading Engine Diagnostic Tool")
    print("This will deep-dive into enhanced_trading_engine.py hanging issues\n")
    
    try:
        timeout = int(input("Enter timeout per test in seconds (default 5): ") or "5")
    except ValueError:
        timeout = 5
    
    diagnostic = EnhancedEngineDiagnostic(timeout_seconds=timeout)
    diagnostic.run_comprehensive_diagnostic()

if __name__ == "__main__":
    main()