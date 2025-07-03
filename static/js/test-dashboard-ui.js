// test-dashboard-ui.js
console.log('[test-dashboard-ui.js] TOP: script loaded');

(function () {
    /**
     * Utility to simulate a click event on a DOM element.
     * @param {HTMLElement} el - The element to click.
     */
    function simulateClick(el) {
        if (!el) return;
        el.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    }

    /**
     * Utility to simulate a change/select event on a DOM element.
     * @param {HTMLSelectElement|HTMLInputElement} el - The element to interact with.
     * @param {string|number} value - The value to set.
     */
    function simulateChange(el, value) {
        if (!el) return;
        el.value = value;
        el.dispatchEvent(new Event('change', { bubbles: true }));
    }

    /**
     * Utility to log results.
     * @param {string} description
     * @param {boolean} passed
     */
    function logResult(description, passed, el) {
        const style = `color: ${passed ? 'green' : 'red'}; font-weight: bold;`;
        console.log(`%c[${passed ? 'PASS' : 'FAIL'}] ${description}`, style);
        if (!passed && el) {
            console.log('Element:', el);
        }
    }

    /**
     * Test all buttons and dropdowns by their IDs or classes.
     */
    function testAllButtonsAndDropdowns() {
        // List all button IDs and dropdown IDs you want to test here
        const buttonIds = [
            // Overview Quick Actions
            'start-trading', 'pause-trading', 'stop-trading', 'deploy-strategy', 'emergency-stop',
            // ML Training
            'start-training', 'refresh-models',
            // Market
            'refresh-market',
            // Chat
            'send-chat',
            // Settings
            'save-settings-btn', 'update-api-keys-btn',
            // Add more as needed...
        ];

        const dropdownIds = [
            'strategy-select', 'model-select', 'market-currency', 'theme-select'
        ];

        // Test buttons
        buttonIds.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                try {
                    simulateClick(el);
                    logResult(`Button #${id} clicked`, true);
                } catch (e) {
                    logResult(`Button #${id} click failed: ${e}`, false, el);
                }
            } else {
                logResult(`Button #${id} not found`, false);
            }
        });

        // Test dropdowns (try changing to the second option if exists)
        dropdownIds.forEach(id => {
            const el = document.getElementById(id);
            if (el && el.options && el.options.length > 1) {
                try {
                    const oldValue = el.value;
                    const newValue = el.options[1].value;
                    simulateChange(el, newValue);
                    logResult(`Dropdown #${id} changed to "${newValue}"`, true);
                    // Change back to old value
                    simulateChange(el, oldValue);
                } catch (e) {
                    logResult(`Dropdown #${id} change failed: ${e}`, false, el);
                }
            } else if (!el) {
                logResult(`Dropdown #${id} not found`, false);
            } else {
                logResult(`Dropdown #${id} has insufficient options`, false, el);
            }
        });

        // Optionally test all elements with class 'btn'
        const allBtns = document.querySelectorAll('button.btn');
        allBtns.forEach(btn => {
            try {
                simulateClick(btn);
                logResult(`Generic button clicked [text="${btn.textContent.trim()}"]`, true);
            } catch (e) {
                logResult(`Generic button click failed [text="${btn.textContent.trim()}"]: ${e}`, false, btn);
            }
        });

        // Optionally test all elements with class 'form-select'
        const allSelects = document.querySelectorAll('select.form-select');
        allSelects.forEach(sel => {
            if (sel.options.length > 1) {
                try {
                    const oldValue = sel.value;
                    const newValue = sel.options[1].value;
                    simulateChange(sel, newValue);
                    logResult(`Generic select changed [id="${sel.id}" to "${newValue}"]`, true);
                    simulateChange(sel, oldValue);
                } catch (e) {
                    logResult(`Generic select change failed [id="${sel.id}"]: ${e}`, false, sel);
                }
            }
        });

        console.log('%cUI Button & Dropdown Test Complete', 'color: blue; font-weight: bold; font-size: 1.2em;');
    }

    // Always export to window for manual use in the console
    window.testAllButtonsAndDropdowns = testAllButtonsAndDropdowns;
    console.log('[test-dashboard-ui.js] BOTTOM: exported testAllButtonsAndDropdowns, typeof:', typeof window.testAllButtonsAndDropdowns);
})();