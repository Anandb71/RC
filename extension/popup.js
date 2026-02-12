/**
 * RC-Oracle — Popup Script
 * Checks backend server connection and displays status.
 */

(function () {
    'use strict';

    const urlInput = document.getElementById('backend-url');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const geminiDot = document.getElementById('gemini-dot');
    const geminiText = document.getElementById('gemini-text');

    async function checkStatus() {
        const url = urlInput.value.trim();

        try {
            const resp = await fetch(`${url}/`, { method: 'GET' });
            const data = await resp.json();

            statusDot.className = 'dot online';
            statusText.textContent = 'Online';

            if (data.ai === 'connected') {
                geminiDot.className = 'dot online';
                geminiText.textContent = 'Connected';
            } else {
                geminiDot.className = 'dot offline';
                geminiText.textContent = 'No API Key';
            }
        } catch {
            statusDot.className = 'dot offline';
            statusText.textContent = 'Offline';
            geminiDot.className = 'dot offline';
            geminiText.textContent = '—';
        }
    }

    // Check on load
    checkStatus();

    // Re-check when URL changes
    urlInput.addEventListener('change', checkStatus);

    // Periodic check
    setInterval(checkStatus, 10000);
})();
