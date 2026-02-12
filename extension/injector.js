/**
 * RC-Oracle — MAIN World Injector (Fully Dynamic)
 * Tries Monaco → CodeMirror → Ace → plain textarea.
 * Communicates with content script via window.postMessage.
 */

(function () {
    'use strict';

    window.__rcOracle = window.__rcOracle || {};

    // ─── Editor Detection ────────────────────────────────────────────────

    /**
     * Try all known editor APIs and return { getCode, setCode } or null.
     */
    function detectEditor() {
        // 1. Monaco
        try {
            if (window.monaco && window.monaco.editor) {
                const models = window.monaco.editor.getModels();
                if (models && models.length > 0) {
                    const model = models[0];
                    return {
                        name: 'Monaco',
                        getCode: () => model.getValue(),
                        setCode: (c) => { model.setValue(c); return true; },
                    };
                }
            }
        } catch (e) { /* skip */ }

        // 2. CodeMirror 6 (via EditorView on DOM)
        try {
            const cmEl = document.querySelector('.cm-editor');
            if (cmEl && cmEl.cmView && cmEl.cmView.view) {
                const view = cmEl.cmView.view;
                return {
                    name: 'CodeMirror6',
                    getCode: () => view.state.doc.toString(),
                    setCode: (c) => {
                        view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: c } });
                        return true;
                    },
                };
            }
        } catch (e) { /* skip */ }

        // 3. CodeMirror 5
        try {
            const cmta = document.querySelector('.CodeMirror');
            if (cmta && cmta.CodeMirror) {
                const cm = cmta.CodeMirror;
                return {
                    name: 'CodeMirror5',
                    getCode: () => cm.getValue(),
                    setCode: (c) => { cm.setValue(c); return true; },
                };
            }
        } catch (e) { /* skip */ }

        // 4. Ace Editor
        try {
            const aceEl = document.querySelector('.ace_editor');
            if (aceEl && aceEl.env && aceEl.env.editor) {
                const editor = aceEl.env.editor;
                return {
                    name: 'Ace',
                    getCode: () => editor.getValue(),
                    setCode: (c) => { editor.setValue(c, -1); return true; },
                };
            }
        } catch (e) { /* skip */ }

        // 5. Fallback: find any large textarea that looks like a code editor
        try {
            const textareas = Array.from(document.querySelectorAll('textarea'));
            for (const ta of textareas) {
                const rect = ta.getBoundingClientRect();
                // Code editors are usually large textareas
                if (rect.width > 200 && rect.height > 100) {
                    const text = (ta.value || '').trim();
                    const ph = (ta.placeholder || '').toLowerCase();
                    const cls = (ta.className || '').toLowerCase();
                    const name = (ta.name || '').toLowerCase();
                    // Skip input/output textareas
                    if (ph.includes('output') || name.includes('output') || name.includes('custom-output')) continue;
                    if (ph.includes('input') && !ph.includes('code')) continue;
                    // Prefer textareas with code-like attributes
                    const isCodeLike = cls.includes('code') || cls.includes('editor') ||
                        ph.includes('code') || ph.includes('solution') ||
                        name.includes('code') || name.includes('editor') ||
                        text.includes('def ') || text.includes('function ') ||
                        text.includes('#include') || text.includes('import ');
                    if (isCodeLike || rect.height > 200) {
                        const nativeSetter = Object.getOwnPropertyDescriptor(
                            window.HTMLTextAreaElement.prototype, 'value'
                        ).set;
                        return {
                            name: 'Textarea',
                            getCode: () => ta.value,
                            setCode: (c) => {
                                nativeSetter.call(ta, c);
                                ta.dispatchEvent(new Event('input', { bubbles: true }));
                                ta.dispatchEvent(new Event('change', { bubbles: true }));
                                return true;
                            },
                        };
                    }
                }
            }
        } catch (e) { /* skip */ }

        return null;
    }

    // ─── Public API ──────────────────────────────────────────────────────

    window.__rcOracle.getCode = function () {
        const editor = detectEditor();
        return editor ? editor.getCode() : null;
    };

    window.__rcOracle.injectCode = function (code) {
        const editor = detectEditor();
        if (editor) {
            console.log(`[RC-Oracle] Injecting via ${editor.name}`);
            return editor.setCode(code);
        }
        return false;
    };

    // ─── Message Listener with Retry ─────────────────────────────────────

    window.addEventListener('message', function (event) {
        if (event.source !== window) return;
        const msg = event.data;
        if (!msg || msg.source !== 'rc-oracle-content') return;

        switch (msg.action) {
            case 'getCode': {
                const code = window.__rcOracle.getCode();
                window.postMessage({
                    source: 'rc-oracle-injector',
                    action: 'codeResult',
                    code: code,
                }, '*');
                break;
            }
            case 'injectCode': {
                let success = window.__rcOracle.injectCode(msg.code);

                // If injection failed, retry a few times (editor may still be loading)
                if (!success) {
                    let retries = 0;
                    const retryInterval = setInterval(() => {
                        success = window.__rcOracle.injectCode(msg.code);
                        retries++;
                        if (success || retries >= 5) {
                            clearInterval(retryInterval);
                            window.postMessage({
                                source: 'rc-oracle-injector',
                                action: 'injectResult',
                                success: success,
                            }, '*');
                        }
                    }, 500);
                } else {
                    window.postMessage({
                        source: 'rc-oracle-injector',
                        action: 'injectResult',
                        success: true,
                    }, '*');
                }
                break;
            }
        }
    });

    console.log('[RC-Oracle] Injector bridge loaded (dynamic editor detection).');
})();
