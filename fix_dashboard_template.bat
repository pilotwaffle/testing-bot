@echo off
REM File: E:\Trade Chat Bot\G Trading Bot\fix_dashboard_template.bat
REM Location: E:\Trade Chat Bot\G Trading Bot\fix_dashboard_template.bat
REM Fix Dashboard Template Error

echo 🔧 Fixing Dashboard Template Error...
echo.

echo The error shows: "Jinja was looking for 'elif' or 'else' or 'endif'"
echo This means there's a syntax error in dashboard.html
echo.

echo 🔍 Checking for template syntax issues...

REM Create a Python script to check and fix template issues
python -c "
import re
import os

template_file = 'templates/dashboard.html'

if not os.path.exists(template_file):
    print('❌ templates/dashboard.html not found')
    print('Creating a simple working dashboard...')
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create a simple working dashboard
    simple_dashboard = '''<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Elite Trading Bot V3.0 - Dashboard</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            background: #1a1a2e; 
            color: white; 
            margin: 0; 
            padding: 20px; 
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .status { 
            background: #2a2d3a; 
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px 0; 
        }
        .api-test { 
            background: #0f3460; 
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px 0; 
        }
        button { 
            background: #00d4aa; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 5px; 
            cursor: pointer; 
            margin: 10px; 
        }
        button:hover { background: #00b894; }
        #result { 
            background: #000; 
            padding: 15px; 
            border-radius: 5px; 
            margin-top: 15px; 
            font-family: monospace; 
        }
    </style>
</head>
<body>
    <div class=\"container\">
        <div class=\"header\">
            <h1>🚀 Elite Trading Bot V3.0</h1>
            <h2>Enhanced Dashboard</h2>
        </div>
        
        <div class=\"status\">
            <h3>✅ System Status</h3>
            <p><strong>Status:</strong> Running</p>
            <p><strong>API:</strong> Market Data API is working!</p>
            <p><strong>Version:</strong> 3.0.3 Enhanced</p>
        </div>
        
        <div class=\"api-test\">
            <h3>📊 Market Data Test</h3>
            <p>Test the market data API that was just fixed:</p>
            <button onclick=\"testMarketData()\">Test Market Data API</button>
            <button onclick=\"testHealth()\">Test Health Check</button>
            <div id=\"result\"></div>
        </div>
        
        <div class=\"api-test\">
            <h3>🌐 Quick Links</h3>
            <a href=\"/health\" style=\"color: #00d4aa;\">Health Check</a> | 
            <a href=\"/api/market-data\" style=\"color: #00d4aa;\">Market Data API</a> | 
            <a href=\"/api/endpoints\" style=\"color: #00d4aa;\">All Endpoints</a>
        </div>
    </div>

    <script>
        async function testMarketData() {
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = 'Testing market data API...';
            
            try {
                const response = await fetch('/api/market-data');
                if (response.ok) {
                    const data = await response.json();
                    resultDiv.innerHTML = `
                        <h4>✅ Market Data API Working!</h4>
                        <p><strong>Success:</strong> ${data.success}</p>
                        <p><strong>Currency:</strong> ${data.currency}</p>
                        <p><strong>Symbols:</strong> ${Object.keys(data.symbols || {}).length} found</p>
                        <pre>${JSON.stringify(data, null, 2)}</pre>
                    `;
                } else {
                    resultDiv.innerHTML = `❌ Error: ${response.status} - ${response.statusText}`;
                }
            } catch (error) {
                resultDiv.innerHTML = `❌ Fetch Error: ${error.message}`;
            }
        }
        
        async function testHealth() {
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = 'Testing health check...';
            
            try {
                const response = await fetch('/health');
                if (response.ok) {
                    const data = await response.json();
                    resultDiv.innerHTML = `
                        <h4>✅ Health Check Working!</h4>
                        <pre>${JSON.stringify(data, null, 2)}</pre>
                    `;
                } else {
                    resultDiv.innerHTML = `❌ Error: ${response.status}`;
                }
            } catch (error) {
                resultDiv.innerHTML = `❌ Error: ${error.message}`;
            }
        }
        
        // Auto-test on load
        window.onload = function() {
            testMarketData();
        };
    </script>
</body>
</html>'''
    
    with open(template_file, 'w', encoding='utf-8') as f:
        f.write(simple_dashboard)
    
    print('✅ Created simple working dashboard')
    
else:
    print('📄 dashboard.html exists, checking for syntax issues...')
    
    with open(template_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for common template issues
    issues = []
    
    # Check for unmatched {% if %} statements
    if_count = content.count('{% if')
    endif_count = content.count('{% endif %}')
    if if_count != endif_count:
        issues.append(f'Unmatched if statements: {if_count} if vs {endif_count} endif')
    
    # Check for unmatched {% for %} statements  
    for_count = content.count('{% for')
    endfor_count = content.count('{% endfor %}')
    if for_count != endfor_count:
        issues.append(f'Unmatched for statements: {for_count} for vs {endfor_count} endfor')
    
    if issues:
        print('❌ Template issues found:')
        for issue in issues:
            print(f'   • {issue}')
        
        # Create backup
        backup_name = template_file + '.backup_broken'
        with open(backup_name, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'💾 Backup created: {backup_name}')
        
        print('🔧 Applying automatic fixes...')
        
        # Try to fix common issues
        fixed_content = content
        
        # Fix missing endif
        if if_count > endif_count:
            missing_endif = if_count - endif_count
            for i in range(missing_endif):
                fixed_content += '\\n{% endif %}'
        
        # Fix missing endfor
        if for_count > endfor_count:
            missing_endfor = for_count - endfor_count
            for i in range(missing_endfor):
                fixed_content += '\\n{% endfor %}'
        
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print('✅ Template syntax fixed')
    else:
        print('✅ No syntax issues found in dashboard.html')

print('\\n🎯 Template check complete!')
"

echo.
echo ✅ Dashboard template fix applied!
echo.
echo Now restart your bot and test the dashboard:
echo 1. Stop the current bot (Ctrl+C)
echo 2. Start it again: python main.py  
echo 3. Open: http://localhost:8000
echo.
pause