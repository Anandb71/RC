/**
 * RC-Oracle — MAIN World Injector
 * Accesses window.monaco.editor to read/write the code editor.
 * Communicates with content script via window.postMessage.
 */

(function () {
    'use strict';

    // Namespace for RC-Oracle bridge
    window.__rcOracle = window.__rcOracle || {};

    /**
     * Get the primary Monaco editor model.
     */
    window.__rcOracle.getModel = function () {
        try {
            if (window.monaco && window.monaco.editor) {
                const models = window.monaco.editor.getModels();
                if (models && models.length > 0) {
                    return models[0];
                }
            }
        } catch (e) {
            console.warn('[RC-Oracle] Monaco not available:', e);
        }
        return null;
    };

    /**
     * Get current editor content.
     */
    window.__rcOracle.getCode = function () {
        const model = window.__rcOracle.getModel();
        return model ? model.getValue() : null;
    };

    /**
     * Inject code into the Monaco editor.
     */
    window.__rcOracle.injectCode = function (code) {
        const model = window.__rcOracle.getModel();
        if (model) {
            model.setValue(code);
            return true;
        }
        return false;
    };

    // Listen for messages from content script
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
                const success = window.__rcOracle.injectCode(msg.code);
                window.postMessage({
                    source: 'rc-oracle-injector',
                    action: 'injectResult',
                    success: success,
                }, '*');
                break;
            }
        }
    });

    console.log('[RC-Oracle] Injector bridge loaded.');
})();
