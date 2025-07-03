# fix_dashboard_html.py
"""
Script to fix the <script src> path for test-dashboard-ui.js in dashboard.html.
It replaces any relative path (e.g., ../static/js/test-dashboard-ui.js)
with the correct absolute path: /static/js/test-dashboard-ui.js
"""

import re

DASHBOARD_HTML = "templates/dashboard.html"

with open(DASHBOARD_HTML, "r", encoding="utf-8") as f:
    content = f.read()

# Replace any test-dashboard-ui.js script path with the correct one
fixed_content = re.sub(
    r'<script\s+src=["\'](\.\./)?static/js/test-dashboard-ui\.js["\']\s*>\s*</script>',
    '<script src="/static/js/test-dashboard-ui.js"></script>',
    content
)

if content != fixed_content:
    with open(DASHBOARD_HTML, "w", encoding="utf-8") as f:
        f.write(fixed_content)
    print("dashboard.html fixed: test-dashboard-ui.js script tag updated to absolute path.")
else:
    print("No changes made. Script tag already correct or not found.")