/**
 * RC-Oracle — Content Script
 * Stealth Scraper + Anti-Detection + Debug Overlay HUD
 * Tuned for the Yantra/INCRDECR coding challenge portal.
 */

(function () {
    'use strict';

    const BACKEND_URL = 'http://localhost:8000';

    // ═══════════════════════════════════════════════════════════════════════════
    // ANTI-DETECTION — Override visibility & focus APIs
    // ═══════════════════════════════════════════════════════════════════════════

    try {
        Object.defineProperty(document, 'visibilityState', {
            get: () => 'visible',
            configurable: true,
        });

        Object.defineProperty(document, 'hidden', {
            get: () => false,
            configurable: true,
        });

        document.hasFocus = () => true;

        // Block visibilitychange events from firing
        const originalAddEventListener = document.addEventListener.bind(document);
        document.addEventListener = function (type, listener, options) {
            if (type === 'visibilitychange') return;
            return originalAddEventListener(type, listener, options);
        };

        const originalWindowAddEventListener = window.addEventListener.bind(window);
        window.addEventListener = function (type, listener, options) {
            if (type === 'visibilitychange' || type === 'blur' || type === 'focus') return;
            return originalWindowAddEventListener(type, listener, options);
        };

        console.log('[RC-Oracle] Anti-detection active.');
    } catch (e) {
        console.warn('[RC-Oracle] Anti-detection partial failure:', e);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // INJECT MAIN-WORLD SCRIPT (Monaco Bridge)
    // ═══════════════════════════════════════════════════════════════════════════

    function injectMainWorldScript() {
        const script = document.createElement('script');
        script.src = chrome.runtime.getURL('injector.js');
        script.onload = () => script.remove();
        (document.head || document.documentElement).appendChild(script);
    }
    injectMainWorldScript();

    // ═══════════════════════════════════════════════════════════════════════════
    // DYNAMIC ELEMENT DISCOVERY — Heuristic-scored, survives any UI changes
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Find a textarea by scoring ALL textareas on the page using multiple signals.
     * @param {'input'|'output'} role - What role to look for
     * @returns {HTMLTextAreaElement|null}
     */
    function findTextarea(role) {
        const textareas = Array.from(document.querySelectorAll('textarea'));
        if (textareas.length === 0) return null;

        const inputKeywords = ['input', 'stdin', 'custom-input', 'enter', 'type here', 'your input'];
        const outputKeywords = ['output', 'stdout', 'custom-output', 'result', 'answer', 'output here'];
        const keywords = role === 'input' ? inputKeywords : outputKeywords;

        let best = null;
        let bestScore = -1;

        for (const ta of textareas) {
            // Skip our own overlay textareas
            if (ta.closest('#rc-oracle-overlay')) continue;
            let score = 0;

            // Signal 1: placeholder text
            const ph = (ta.placeholder || '').toLowerCase();
            for (const kw of keywords) { if (ph.includes(kw)) score += 10; }

            // Signal 2: name attribute
            const name = (ta.name || '').toLowerCase();
            for (const kw of keywords) { if (name.includes(kw)) score += 10; }

            // Signal 3: id attribute
            const id = (ta.id || '').toLowerCase();
            for (const kw of keywords) { if (id.includes(kw)) score += 10; }

            // Signal 4: aria-label
            const aria = (ta.getAttribute('aria-label') || '').toLowerCase();
            for (const kw of keywords) { if (aria.includes(kw)) score += 8; }

            // Signal 5: class names
            const cls = (ta.className || '').toLowerCase();
            for (const kw of keywords) { if (cls.includes(kw)) score += 5; }

            // Signal 6: nearest label text
            const labelEl = ta.closest('label') || (ta.id && document.querySelector(`label[for="${ta.id}"]`));
            if (labelEl) {
                const labelText = labelEl.textContent.toLowerCase();
                for (const kw of keywords) { if (labelText.includes(kw)) score += 8; }
            }

            // Signal 7: nearby sibling/parent text (within 2 levels up)
            const container = ta.parentElement;
            if (container) {
                const containerText = container.textContent.toLowerCase().slice(0, 200);
                for (const kw of keywords) { if (containerText.includes(kw)) score += 3; }
                const grandParent = container.parentElement;
                if (grandParent) {
                    const gpText = grandParent.textContent.toLowerCase().slice(0, 300);
                    for (const kw of keywords) { if (gpText.includes(kw)) score += 2; }
                }
            }

            // Signal 8: data-* attributes
            for (const attr of ta.attributes) {
                if (attr.name.startsWith('data-')) {
                    const val = attr.value.toLowerCase();
                    for (const kw of keywords) { if (val.includes(kw)) score += 7; }
                }
            }

            // Signal 9: readonly/disabled = more likely output
            if (role === 'output' && (ta.readOnly || ta.disabled)) score += 5;
            if (role === 'input' && !ta.readOnly && !ta.disabled) score += 3;

            // Signal 10: visible and has reasonable size
            const rect = ta.getBoundingClientRect();
            if (rect.width > 50 && rect.height > 20) score += 2;

            if (score > bestScore) {
                bestScore = score;
                best = ta;
            }
        }

        return best;
    }

    /**
     * Find the Run/Execute button using heuristic scoring.
     * @returns {HTMLButtonElement|null}
     */
    function findRunButton() {
        const buttons = Array.from(document.querySelectorAll('button, input[type="button"], input[type="submit"], [role="button"]'));
        const runKeywords = ['run', 'execute', 'submit', 'compile', 'test'];
        const excludeKeywords = ['login', 'signup', 'register', 'save', 'delete', 'logout', 'upload', 'download'];

        let best = null;
        let bestScore = -1;

        for (const btn of buttons) {
            // Skip our own buttons
            if (btn.closest('#rc-oracle-overlay')) continue;
            let score = 0;

            const text = (btn.textContent || btn.value || '').trim().toLowerCase();

            // Exact match is strongest
            if (text === 'run' || text === 'run code') score += 20;
            if (text === 'execute') score += 15;

            // Partial match
            for (const kw of runKeywords) { if (text.includes(kw)) score += 8; }

            // Exclude unrelated buttons
            for (const kw of excludeKeywords) { if (text.includes(kw)) score -= 20; }

            // Check data-action, title, aria-label
            const action = (btn.getAttribute('data-action') || '').toLowerCase();
            const title = (btn.title || '').toLowerCase();
            const ariaL = (btn.getAttribute('aria-label') || '').toLowerCase();
            for (const attr of [action, title, ariaL]) {
                for (const kw of runKeywords) { if (attr.includes(kw)) score += 10; }
            }

            // Class names
            const cls = (btn.className || '').toLowerCase();
            for (const kw of runKeywords) { if (cls.includes(kw)) score += 5; }

            // Icon buttons: look for play icon (▶, ►) or SVG with play-like path
            if (text.includes('▶') || text.includes('►')) score += 12;
            if (btn.querySelector('svg')) {
                const svgText = btn.innerHTML.toLowerCase();
                if (svgText.includes('play') || svgText.includes('triangle')) score += 8;
            }

            // Proximity: is it near a textarea? (likely the terminal area)
            const rect = btn.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                score += 1; // visible
                const nearbyTextarea = btn.closest('div, section, form')?.querySelector('textarea');
                if (nearbyTextarea) score += 5;
            }

            // Green/primary colored buttons are more likely "run"
            const style = getComputedStyle(btn);
            const bg = style.backgroundColor;
            if (bg.includes('0, 128') || bg.includes('76, 175') || bg.includes('40, 167') || bg.includes('success')) {
                score += 3;
            }

            if (score > bestScore) {
                bestScore = score;
                best = btn;
            }
        }

        return best;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // STEALTH SCRAPER — Extract I/O from the portal page
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Scrape I/O pairs from the current page.
     * Fully dynamic — uses heuristic discovery, regex patterns, and table parsing.
     */
    function scrapeIOPairs() {
        const pairs = [];

        // ─── Strategy 1: Heuristic textarea discovery ────────────────────
        const customInput = findTextarea('input');
        const customOutput = findTextarea('output');
        if (customInput && customOutput && customInput !== customOutput &&
            customInput.value.trim() && customOutput.value.trim()) {
            pairs.push({
                input: customInput.value.trim(),
                output: customOutput.value.trim(),
            });
        }

        // ─── Strategy 2: Multi-line Example blocks (competitive programming) ──
        // Capture everything between "Input:" and "Output:" and between "Output:" and next section
        const bodyText = document.body.innerText;
        const exampleBlocks = bodyText.split(/Example\s*:?/gi).slice(1);

        for (const block of exampleBlocks) {
            // Find Input: ... Output: ... pattern (multi-line)
            const ioMatch = block.match(/Input\s*:\s*\n([\s\S]*?)\s*Output\s*:\s*\n([\s\S]*?)(?:\n\s*(?:Explanation|Constraints|Example|Note|$))/i);
            if (ioMatch) {
                const inp = ioMatch[1].trim();
                const out = ioMatch[2].trim();
                if (inp && out) {
                    const exists = pairs.some(p => p.input === inp && p.output === out);
                    if (!exists) pairs.push({ input: inp, output: out });
                }
                continue;
            }

            // Fallback: simpler pattern
            const simpleMatch = block.match(/Input\s*:\s*\n?\s*(.+?)\s*Output\s*:\s*\n?\s*(.+?)(?:\n|$)/i);
            if (simpleMatch) {
                const inp = simpleMatch[1].trim();
                const out = simpleMatch[2].trim();
                if (inp && out) {
                    const exists = pairs.some(p => p.input === inp && p.output === out);
                    if (!exists) pairs.push({ input: inp, output: out });
                }
            }
        }

        // ─── Strategy 3: Global Input/Output pattern (if no Example blocks found)
        if (pairs.length === 0) {
            // Try to find standalone Input:/Output: sections
            const globalMatch = bodyText.match(/Input\s*:\s*\n([\s\S]*?)\s*Output\s*:\s*\n([\s\S]*?)(?:\n\s*(?:Explanation|Constraints|Run|$))/i);
            if (globalMatch) {
                const inp = globalMatch[1].trim();
                const out = globalMatch[2].trim();
                if (inp && out) pairs.push({ input: inp, output: out });
            }
        }

        // ─── Strategy 4: Tables with input/output columns ───────────────
        const tables = document.querySelectorAll('table');
        for (const table of tables) {
            const headers = Array.from(table.querySelectorAll('th')).map(th =>
                th.textContent.trim().toLowerCase()
            );
            const inputIdx = headers.findIndex(h => h.includes('input'));
            const outputIdx = headers.findIndex(h => h.includes('output'));

            if (inputIdx !== -1 && outputIdx !== -1) {
                const rows = table.querySelectorAll('tbody tr, tr:not(:first-child)');
                for (const row of rows) {
                    const cells = row.querySelectorAll('td');
                    if (cells.length > Math.max(inputIdx, outputIdx)) {
                        const inp = cells[inputIdx].textContent.trim();
                        const out = cells[outputIdx].textContent.trim();
                        if (inp && out) {
                            const exists = pairs.some(p => p.input === inp && p.output === out);
                            if (!exists) pairs.push({ input: inp, output: out });
                        }
                    }
                }
            }
        }

        // ─── Strategy 5: pre/code blocks with Input/Output labels ────────
        if (pairs.length === 0) {
            const preBlocks = document.querySelectorAll('pre, code');
            const blockArray = Array.from(preBlocks);
            for (let i = 0; i < blockArray.length - 1; i++) {
                const el = blockArray[i];
                const nextEl = blockArray[i + 1];
                // Check if preceding text contains "Input" and next has "Output"
                const prevText = el.previousSibling?.textContent?.trim() ||
                    el.parentElement?.previousElementSibling?.textContent?.trim() || '';
                const nextPrevText = nextEl.previousSibling?.textContent?.trim() ||
                    nextEl.parentElement?.previousElementSibling?.textContent?.trim() || '';
                if (/input/i.test(prevText) && /output/i.test(nextPrevText)) {
                    const inp = el.textContent.trim();
                    const out = nextEl.textContent.trim();
                    if (inp && out) {
                        const exists = pairs.some(p => p.input === inp && p.output === out);
                        if (!exists) pairs.push({ input: inp, output: out });
                    }
                }
            }
        }

        // ─── Strategy 6: Div-based I/O containers ────────────────────────
        if (pairs.length === 0) {
            const allDivs = document.querySelectorAll('div, section, article');
            for (const div of allDivs) {
                const heading = div.querySelector('h1, h2, h3, h4, h5, h6, strong, b, span');
                if (!heading) continue;
                const headText = heading.textContent.trim().toLowerCase();
                if (headText.includes('sample') || headText.includes('example') || headText.includes('test case')) {
                    const text = div.textContent;
                    const ioMatch = text.match(/Input\s*:?\s*\n?([\s\S]*?)\s*Output\s*:?\s*\n?([\s\S]*?)(?:\n\s*(?:Explanation|Note|$))/i);
                    if (ioMatch) {
                        const inp = ioMatch[1].trim();
                        const out = ioMatch[2].trim();
                        if (inp && out) {
                            const exists = pairs.some(p => p.input === inp && p.output === out);
                            if (!exists) pairs.push({ input: inp, output: out });
                        }
                    }
                }
            }
        }

        return pairs;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MUTATION OBSERVER — Detect new questions (DEBOUNCED to avoid lag)
    // ═══════════════════════════════════════════════════════════════════════════

    let observerTimer = null;

    const observer = new MutationObserver(() => {
        // Debounce: wait 2 seconds of DOM silence before checking
        if (observerTimer) clearTimeout(observerTimer);
        observerTimer = setTimeout(() => {
            try {
                const pairs = scrapeIOPairs();
                if (pairs.length > 0) {
                    log(`Auto-scraped ${pairs.length} I/O pair(s)`, 'info');
                    window.__rcOraclePairs = pairs;
                }
            } catch (e) {
                // Silently ignore scrape errors
            }
        }, 2000);
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true,
        // DO NOT observe characterData — it fires on every keystroke and causes lag
    });

    // ═══════════════════════════════════════════════════════════════════════════
    // DEBUG OVERLAY — Draggable HUD
    // ═══════════════════════════════════════════════════════════════════════════

    function createOverlay() {
        const overlay = document.createElement('div');
        overlay.id = 'rc-oracle-overlay';
        overlay.innerHTML = `
      <div id="rc-header">
        <div class="rc-title">
          <span class="rc-logo">⚡</span>
          <span>RC-Oracle</span>
          <span class="rc-status-dot" id="rc-status-dot"></span>
        </div>
        <button id="rc-minimize-btn" title="Minimize">─</button>
      </div>
      <div id="rc-body">
        <div id="rc-logic-section">
          <div class="rc-section-label">Currently Inferred Logic</div>
          <div id="rc-logic-display">Waiting for analysis...</div>
        </div>

        <button id="rc-examine-btn">⚡ EXAMINE & INJECT</button>

        <div id="rc-lang-section">
          <div class="rc-section-label">Inject Language</div>
          <div id="rc-lang-toggle">
            <button class="rc-lang-btn active" data-lang="cpp">C++</button>
            <button class="rc-lang-btn" data-lang="python">Python</button>
          </div>
        </div>

        <div id="rc-bulk-section">
          <div class="rc-section-label">Bulk Data Upload</div>
          <textarea
            id="rc-bulk-textarea"
            placeholder='Paste JSON: [{"input":"3", "output":"10"}, ...]'
          ></textarea>
          <button id="rc-bulk-upload-btn">📤 Upload & Solve</button>
        </div>

        <div id="rc-log-section">
          <div class="rc-section-label">Pipeline Log</div>
          <div id="rc-log"></div>
        </div>
      </div>
    `;

        document.body.appendChild(overlay);
        setupDragging(overlay);
        setupActions();
        checkServerConnection();
    }

    // ─── Dragging ─────────────────────────────────────────────────────────────

    function setupDragging(overlay) {
        const header = overlay.querySelector('#rc-header');
        let isDragging = false;
        let startX, startY, initialX, initialY;

        header.addEventListener('mousedown', (e) => {
            if (e.target.id === 'rc-minimize-btn') return;
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            const rect = overlay.getBoundingClientRect();
            initialX = rect.left;
            initialY = rect.top;
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            overlay.style.left = initialX + dx + 'px';
            overlay.style.top = initialY + dy + 'px';
            overlay.style.right = 'auto';
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
        });
    }

    // ─── Actions ──────────────────────────────────────────────────────────────

    let selectedLang = 'cpp';

    function setupActions() {
        // Minimize toggle
        document.getElementById('rc-minimize-btn').addEventListener('click', () => {
            const overlay = document.getElementById('rc-oracle-overlay');
            overlay.classList.toggle('minimized');
            const btn = document.getElementById('rc-minimize-btn');
            btn.textContent = overlay.classList.contains('minimized') ? '□' : '─';
        });

        // Language toggle
        document.querySelectorAll('.rc-lang-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.rc-lang-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                selectedLang = btn.dataset.lang;
                log(`Inject language: ${selectedLang.toUpperCase()}`, 'info');
            });
        });

        // Examine & Inject
        document.getElementById('rc-examine-btn').addEventListener('click', handleExamineAndInject);

        // Bulk Upload
        document.getElementById('rc-bulk-upload-btn').addEventListener('click', handleBulkUpload);
    }

    // ─── Server Check ─────────────────────────────────────────────────────────

    async function checkServerConnection() {
        const dot = document.getElementById('rc-status-dot');
        try {
            const resp = await fetch(`${BACKEND_URL}/`, { signal: AbortSignal.timeout(3000) });
            const data = await resp.json();
            if (data.status === 'online') {
                dot.classList.remove('disconnected');
                log('Backend connected', 'success');
            } else {
                dot.classList.add('disconnected');
                log('Backend status unknown', 'warn');
            }
        } catch {
            dot.classList.add('disconnected');
            log('Backend offline — start server.py', 'error');
        }
    }

    // ─── EXAMINE & INJECT Pipeline ────────────────────────────────────────────

    /**
     * Scrape the problem description and constraints from the page.
     * Returns { description, constraints, minVal, maxVal, inputLines }
     */
    function scrapeDescription() {
        const bodyText = document.body.innerText;
        const result = { description: '', constraints: '', minVal: null, maxVal: null, inputLines: 1 };

        // Extract Test Case Description
        const descMatch = bodyText.match(/Test\s*Case\s*Description\s*:?\s*\n?([\s\S]*?)(?=Constraints|Example|Input|$)/i);
        if (descMatch) result.description = descMatch[1].trim();

        // Extract Constraints section
        const constMatch = bodyText.match(/Constraints\s*:?\s*\n?([\s\S]*?)(?=Example|Input|Output|$)/i);
        if (constMatch) result.constraints = constMatch[1].trim();

        // Parse numeric ranges from constraints like "1 <= n <= 9999" or "1 <= N <= 10^5"
        const rangePatterns = [
            /(-?\d+)\s*<=?\s*\w+\s*<=?\s*(\d+(?:\s*\*?\s*10\s*\^?\s*\d+)?)/g,
            /(\d+)\s*<\s*\w+\s*<\s*(\d+)/g,
        ];
        for (const pattern of rangePatterns) {
            let match;
            while ((match = pattern.exec(result.constraints)) !== null) {
                const lo = parseInt(match[1]);
                let hiStr = match[2].replace(/\s/g, '');
                let hi;
                // Handle scientific notation: 10^5, 10**5
                if (/\d+[\^*]+\d+/.test(hiStr)) {
                    const parts = hiStr.split(/[\^*]+/);
                    hi = Math.pow(parseInt(parts[0]), parseInt(parts[parts.length - 1]));
                } else {
                    hi = parseInt(hiStr);
                }
                if (!isNaN(lo) && (result.minVal === null || lo < result.minVal)) result.minVal = lo;
                if (!isNaN(hi) && (result.maxVal === null || hi > result.maxVal)) result.maxVal = hi;
            }
        }

        // Detect how many lines of input from description
        const desc = (result.description + ' ' + result.constraints).toLowerCase();
        if (desc.includes('second line') || desc.includes('next line') || desc.includes('followed by')) {
            result.inputLines = 2;
        }
        if (desc.includes('third line') || desc.includes('n lines') || desc.includes('next n')) {
            result.inputLines = 3;
        }

        return result;
    }

    /**
     * Detect input format from existing example pairs.
     */
    function detectInputFormat(existingPairs) {
        if (existingPairs.length === 0) return 'unknown';

        const formats = existingPairs.map(p => {
            const lines = p.input.split('\n').map(l => l.trim()).filter(l => l.length > 0);
            if (lines.length === 1) {
                const parts = lines[0].split(/\s+/);
                if (parts.length === 1 && /^-?\d+$/.test(parts[0])) return 'single_int';
                if (parts.every(x => /^-?\d+$/.test(x))) return 'space_separated';
                return 'single_string';
            }
            if (lines.length === 2) {
                const firstParts = lines[0].split(/\s+/);
                const secondParts = lines[1].split(/\s+/);
                if (firstParts.length === 1 && /^\d+$/.test(firstParts[0])) {
                    const n = parseInt(firstParts[0]);
                    if (secondParts.length === n && secondParts.every(x => /^-?\d+$/.test(x))) {
                        return 'n_then_numbers';
                    }
                }
                if (firstParts.length === 1 && secondParts.length === 1 &&
                    /^-?\d+$/.test(firstParts[0]) && /^-?\d+$/.test(secondParts[0])) {
                    return 'two_ints';
                }
                return 'multi_line';
            }
            return 'multi_line';
        });

        const counts = {};
        for (const f of formats) counts[f] = (counts[f] || 0) + 1;
        return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
    }

    /**
     * Generate probe inputs matching the detected format.
     * Uses constraint bounds (minVal/maxVal) from the page description.
     */
    function generateProbeInputs(format, constraints = {}) {
        const lo = constraints.minVal ?? 0;
        const hi = Math.min(constraints.maxVal ?? 1000, 100000); // cap at 100k for safety
        const ri = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
        const ra = (n, min, max) => Array.from({ length: n }, () => ri(min, max)).join(' ');
        const inputs = [];

        // Generate smart boundary values from constraints
        const boundaryVals = [lo, lo + 1, lo + 2, Math.floor((lo + hi) / 2), hi - 1, hi];
        // Add power-of-2 and common edge values within range
        for (const v of [0, 1, 2, 3, 5, 7, 8, 10, 15, 16, 31, 32, 42, 50, 63, 64, 100, 127, 128, 255, 256, 500, 999, 1000, 9999]) {
            if (v >= lo && v <= hi) boundaryVals.push(v);
        }
        const uniqueVals = [...new Set(boundaryVals)].filter(v => v >= lo && v <= hi).sort((a, b) => a - b);

        switch (format) {
            case 'single_int':
                for (const n of uniqueVals) inputs.push(String(n));
                // Add some random values within range
                for (let i = 0; i < 10; i++) inputs.push(String(ri(lo, Math.min(hi, 9999))));
                // Negative values only if lo allows
                if (lo < 0) {
                    for (const n of [-1, -2, -5, -10, -50, -100]) {
                        if (n >= lo) inputs.push(String(n));
                    }
                }
                break;

            case 'n_then_numbers':
                for (const n of [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]) {
                    if (n >= lo && n <= Math.min(hi, 20))
                        inputs.push(`${n}\n${ra(n, Math.max(lo, -100), Math.min(hi, 100))}`);
                }
                for (const n of [1, 3, 5, 8, 10]) {
                    if (n >= lo && n <= Math.min(hi, 20))
                        inputs.push(`${n}\n${ra(n, Math.max(lo, 1), Math.min(hi, 1000))}`);
                }
                inputs.push('1\n0', '1\n1');
                break;

            case 'space_separated':
                for (let c = 1; c <= 10; c++) inputs.push(ra(c, Math.max(lo, -100), Math.min(hi, 100)));
                for (let c = 2; c <= 5; c++) inputs.push(ra(c, Math.max(lo, 1), Math.min(hi, 1000)));
                break;

            case 'two_ints':
                // Use constraint-bounded pairs
                for (const [a, b] of [[lo, lo], [lo, hi], [hi, lo], [lo + 1, lo + 2], [Math.floor(hi / 2), hi]]) {
                    if (a >= lo && a <= hi && b >= lo && b <= hi) inputs.push(`${a}\n${b}`);
                }
                for (let i = 0; i < 10; i++) {
                    inputs.push(`${ri(lo, Math.min(hi, 1000))}\n${ri(lo, Math.min(hi, 1000))}`);
                }
                break;

            case 'single_string':
                inputs.push(...['hello', 'world', 'abc', 'abcdef', 'a', 'ab', 'xyz', 'test', 'abcabc', 'racecar', 'level', 'aabb', 'aaaa', 'abcd']);
                break;

            case 'multi_line':
            case 'unknown':
            default:
                // Try ALL formats as fallback, using constraints
                for (const n of uniqueVals.slice(0, 12)) inputs.push(String(n));
                for (const n of [1, 3, 5, 10]) inputs.push(`${n}\n${ra(n, Math.max(lo, -100), Math.min(hi, 100))}`);
                for (let c = 2; c <= 5; c++) inputs.push(ra(c, Math.max(lo, 1), Math.min(hi, 100)));
                for (const [a, b] of [[lo, lo + 1], [Math.floor(hi / 2), hi]]) inputs.push(`${a}\n${b}`);
                inputs.push('hello', 'abc', 'test');
                break;
        }
        return inputs;
    }

    /**
     * Use the portal's Terminal to probe. Scrapes problem description for constraints,
     * detects input format from examples, generates matching inputs within bounds.
     */
    async function probeOnSite(existingPairs) {
        // Fully dynamic element discovery using heuristic scoring
        const customInput = findTextarea('input');
        const customOutput = findTextarea('output');
        const runBtn = findRunButton();

        if (!customInput || !customOutput || !runBtn) {
            log('Terminal not found — cannot probe on-site', 'warn');
            log(`  Found: Input=${!!customInput}, Output=${!!customOutput}, RunBtn=${!!runBtn}`, 'warn');
            return existingPairs;
        }
        if (customInput === customOutput) {
            log('Input and Output are the same element — cannot probe', 'warn');
            return existingPairs;
        }

        log('On-site probing started...', 'info');

        // Scrape description and constraints from the page
        const desc = scrapeDescription();
        if (desc.description) log(`  Description: ${desc.description.slice(0, 80)}...`, 'info');
        if (desc.constraints) log(`  Constraints: ${desc.constraints.slice(0, 80)}`, 'info');
        if (desc.minVal !== null || desc.maxVal !== null) {
            log(`  Range: [${desc.minVal ?? '?'}, ${desc.maxVal ?? '?'}]`, 'info');
        }

        const format = detectInputFormat(existingPairs);
        log(`  Detected format: ${format}`, 'info');

        const probeInputs = generateProbeInputs(format, desc);
        log(`  Generated ${probeInputs.length} test inputs`, 'info');

        const pairs = [...existingPairs];
        const existingInputs = new Set(pairs.map(p => p.input));
        let successCount = 0;
        let failCount = 0;

        const nativeSetter = Object.getOwnPropertyDescriptor(
            window.HTMLTextAreaElement.prototype, 'value'
        ).set;

        async function runOneProbe(testInput) {
            nativeSetter.call(customInput, testInput);
            customInput.dispatchEvent(new Event('input', { bubbles: true }));
            customInput.dispatchEvent(new Event('change', { bubbles: true }));
            customInput.dispatchEvent(new InputEvent('input', { bubbles: true, data: testInput }));

            nativeSetter.call(customOutput, '');
            customOutput.dispatchEvent(new Event('input', { bubbles: true }));

            runBtn.click();

            let output = '';
            const prev = customOutput.value;
            for (let i = 0; i < 50; i++) {
                await new Promise(r => setTimeout(r, 200));
                const cur = customOutput.value.trim();
                if (cur && cur !== prev &&
                    !cur.includes('...') &&
                    !cur.toLowerCase().includes('running') &&
                    !cur.toLowerCase().includes('compiling')) {
                    output = cur;
                    break;
                }
            }
            return output;
        }

        for (const testInput of probeInputs) {
            if (existingInputs.has(testInput)) continue;
            if (pairs.length >= 15) break;
            if (failCount >= 5 && successCount === 0) {
                log(`  Format "${format}" not working, trying fallback...`, 'warn');
                break;
            }

            try {
                const output = await runOneProbe(testInput);
                if (output) {
                    pairs.push({ input: testInput, output });
                    existingInputs.add(testInput);
                    successCount++;
                    log(`  ${testInput.replace(/\n/g, '\\n')} -> ${output}`, 'info');
                } else {
                    failCount++;
                }
            } catch (e) { failCount++; }
        }

        // Fallback: if primary format failed, try universal
        if (successCount === 0 && format !== 'unknown') {
            log('  Trying universal fallback inputs...', 'warn');
            const fallback = generateProbeInputs('unknown', desc);
            failCount = 0;
            for (const testInput of fallback) {
                if (existingInputs.has(testInput)) continue;
                if (pairs.length >= 15) break;
                if (failCount >= 8) break;
                try {
                    const output = await runOneProbe(testInput);
                    if (output) {
                        pairs.push({ input: testInput, output });
                        existingInputs.add(testInput);
                        successCount++;
                        log(`  ${testInput.replace(/\n/g, '\\n')} -> ${output}`, 'info');
                    } else { failCount++; }
                } catch (e) { failCount++; }
            }
        }

        log(`Probing done: ${pairs.length} total pairs (${successCount} new)`, 'success');
        return pairs;
    }

    // ─── EXAMINE & INJECT Pipeline ────────────────────────────────────────────

    async function handleExamineAndInject() {
        const btn = document.getElementById('rc-examine-btn');
        btn.classList.add('loading');
        btn.textContent = '⏳ Analyzing...';

        try {
            // Step 1: Scrape I/O pairs from page
            log('Scraping page for I/O pairs...', 'info');
            let pairs = scrapeIOPairs();

            // Merge with any previously auto-scraped or bulk-uploaded
            if (window.__rcOraclePairs && window.__rcOraclePairs.length > 0) {
                const existing = new Set(pairs.map(p => JSON.stringify(p)));
                for (const p of window.__rcOraclePairs) {
                    if (!existing.has(JSON.stringify(p))) {
                        pairs.push(p);
                    }
                }
            }

            log(`Scraped ${pairs.length} I/O pair(s) from page`, 'info');

            // Step 2: On-site probing if we have fewer than 3 pairs
            if (pairs.length < 3) {
                btn.textContent = '🔍 Probing...';
                pairs = await probeOnSite(pairs);
            }

            if (pairs.length < 1) {
                log('No I/O pairs found. Use Bulk Upload or paste data.', 'error');
                resetExamineBtn();
                return;
            }

            log(`Total: ${pairs.length} I/O pair(s) ready`, 'success');

            // Step 2.5: Read boilerplate template from editor
            let boilerplate = null;
            try {
                boilerplate = await readEditorCode();
                if (boilerplate && boilerplate.trim()) {
                    log(`Read boilerplate template (${boilerplate.length} chars)`, 'info');
                } else {
                    boilerplate = null;
                    log('No boilerplate template found — standalone mode', 'info');
                }
            } catch {
                log('Could not read editor template', 'warn');
            }

            // Also scrape description for context
            const descData = scrapeDescription();
            const descText = descData.description || '';
            const constraintsText = descData.constraints || '';
            const fullDescription = [descText, constraintsText].filter(Boolean).join('\n\nConstraints:\n');

            // Step 3: Send to solver
            btn.textContent = '🧠 Solving...';
            log('Sending to solver pipeline...', 'info');
            let solveResp;
            try {
                const solveBody = { pairs };
                if (boilerplate) solveBody.boilerplate = boilerplate;
                if (fullDescription.trim()) solveBody.description = fullDescription;

                solveResp = await fetch(`${BACKEND_URL}/solve`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(solveBody),
                    signal: AbortSignal.timeout(300000), // 5 min for 20 retries
                });
            } catch (fetchErr) {
                log(`Backend unreachable: ${fetchErr.message}`, 'error');
                resetExamineBtn();
                return;
            }

            if (!solveResp.ok) {
                const err = await solveResp.json().catch(() => ({ detail: solveResp.statusText }));
                log(`Solver failed: ${err.detail}`, 'error');
                resetExamineBtn();
                return;
            }

            const solveData = await solveResp.json();
            log(`Solved via ${solveData.method} (${solveData.attempts} attempts)`, 'success');

            // Update logic display
            setLogic(solveData.logic);

            let finalCode = solveData.cpp_code;
            log('C++ code ready ✅', 'success');

            // Step 5: Inject into Monaco editor
            injectCode(finalCode);
            log('Code injected into editor! ✅', 'success');
        } catch (e) {
            log(`Pipeline error: ${e.message}`, 'error');
        }

        resetExamineBtn();
    }

    function resetExamineBtn() {
        const btn = document.getElementById('rc-examine-btn');
        btn.classList.remove('loading');
        btn.textContent = '⚡ EXAMINE & INJECT';
    }

    // ─── Bulk Upload Handler ──────────────────────────────────────────────────

    async function handleBulkUpload() {
        const textarea = document.getElementById('rc-bulk-textarea');
        const raw = textarea.value.trim();

        if (!raw) {
            log('Bulk upload field is empty', 'warn');
            return;
        }

        let pairs;

        // Try JSON first
        try {
            const parsed = JSON.parse(raw);
            if (Array.isArray(parsed)) {
                if (!parsed.every(p => p.input !== undefined && p.output !== undefined)) {
                    throw new Error('Missing input/output fields');
                }
                pairs = parsed;
            } else {
                throw new Error('Not an array');
            }
        } catch {
            // Fallback: try parsing as raw text with Input:/Output: blocks
            pairs = [];
            const lines = raw.split('\n');
            let currentInput = null;
            let readingInput = false;
            let readingOutput = false;
            let inputLines = [];
            let outputLines = [];

            for (const line of lines) {
                const trimmed = line.trim();
                if (/^input\s*:/i.test(trimmed)) {
                    // Save previous pair
                    if (inputLines.length > 0 && outputLines.length > 0) {
                        pairs.push({ input: inputLines.join('\n'), output: outputLines.join('\n') });
                    }
                    inputLines = [];
                    outputLines = [];
                    readingInput = true;
                    readingOutput = false;
                    const afterColon = trimmed.replace(/^input\s*:/i, '').trim();
                    if (afterColon) inputLines.push(afterColon);
                } else if (/^output\s*:/i.test(trimmed)) {
                    readingInput = false;
                    readingOutput = true;
                    const afterColon = trimmed.replace(/^output\s*:/i, '').trim();
                    if (afterColon) outputLines.push(afterColon);
                } else if (readingInput) {
                    inputLines.push(line);
                } else if (readingOutput) {
                    outputLines.push(line);
                }
            }
            // Last pair
            if (inputLines.length > 0 && outputLines.length > 0) {
                pairs.push({ input: inputLines.join('\n'), output: outputLines.join('\n') });
            }

            if (pairs.length === 0) {
                log('Could not parse input. Use JSON [{"input":"...","output":"..."}] or Input:/Output: format', 'error');
                return;
            }
        }

        log(`Bulk loaded ${pairs.length} pair(s)`, 'info');

        // Store for the pipeline
        window.__rcOraclePairs = window.__rcOraclePairs || [];
        const existing = new Set(window.__rcOraclePairs.map(p => JSON.stringify(p)));
        for (const p of pairs) {
            if (!existing.has(JSON.stringify(p))) {
                window.__rcOraclePairs.push(p);
            }
        }

        // Auto-trigger solve
        document.getElementById('rc-examine-btn').click();
    }

    // ─── Editor Code Reading & Injection ─────────────────────────────────────

    /**
     * Read current code from the editor (boilerplate template).
     * Returns a promise that resolves with the code string or null.
     */
    function readEditorCode() {
        return new Promise((resolve) => {
            const timeout = setTimeout(() => {
                window.removeEventListener('message', handler);
                resolve(null);
            }, 3000);

            function handler(event) {
                if (event.source !== window) return;
                const msg = event.data;
                if (msg && msg.source === 'rc-oracle-injector' && msg.action === 'codeResult') {
                    clearTimeout(timeout);
                    window.removeEventListener('message', handler);
                    resolve(msg.code || null);
                }
            }

            window.addEventListener('message', handler);
            window.postMessage({ source: 'rc-oracle-content', action: 'getCode' }, '*');
        });
    }

    function injectCode(code) {
        window.postMessage({
            source: 'rc-oracle-content',
            action: 'injectCode',
            code: code,
        }, '*');
    }

    // Listen for inject result from injector.js
    window.addEventListener('message', (event) => {
        if (event.source !== window) return;
        const msg = event.data;
        if (!msg || msg.source !== 'rc-oracle-injector') return;

        if (msg.action === 'injectResult') {
            if (msg.success) {
                log('Monaco injection confirmed ✅', 'success');
            } else {
                log('Monaco not found — copy code manually', 'warn');
            }
        }
    });

    // ─── Utilities ────────────────────────────────────────────────────────────

    function setLogic(text) {
        const el = document.getElementById('rc-logic-display');
        if (el) el.textContent = text;
    }

    function log(message, level = 'info') {
        const logEl = document.getElementById('rc-log');
        if (!logEl) {
            console.log(`[RC-Oracle] ${message}`);
            return;
        }

        const entry = document.createElement('div');
        entry.className = `log-entry ${level}`;
        const time = new Date().toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
        });
        entry.textContent = `[${time}] ${message}`;
        logEl.appendChild(entry);
        logEl.scrollTop = logEl.scrollHeight;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // BOOT
    // ═══════════════════════════════════════════════════════════════════════════

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createOverlay);
    } else {
        createOverlay();
    }

    console.log('[RC-Oracle] Content script loaded.');
})();
