(function() {
    'use strict';

    if (window.__focus_js_v7_initialized__) return;
    window.__focus_js_v7_initialized__ = true;

    // 1. Force light color scheme (no DOM observers, purely stylistic)
    document.documentElement.style.colorScheme = 'light';

    var NARROW_QUERY = window.matchMedia('(max-width: 920px), (max-height: 520px)');

    // 2. Autofocus input textarea after DOM loads (desktop only; avoid iOS keyboard jump on load)
    function focusInput() {
        if (NARROW_QUERY.matches) return;
        var el = document.querySelector('#chat-input textarea') || document.querySelector('textarea');
        if (el) el.focus();
    }
    setTimeout(focusInput, 300);
    setTimeout(focusInput, 1000);

    // 3. Track reasoning-box open/closed state & real-time elapsed timer in top-right corner
    document.addEventListener('toggle', function(event) {
        if (event.target && (event.target.id === 'reasoning-box' || event.target.classList.contains('reasoning-box'))) {
            window.isReasoningBoxOpen = event.target.open;
        }
    }, true);

    setInterval(function() {
        var boxes = document.querySelectorAll('.reasoning-box[data-status="thinking"][data-start-ms]');
        if (!boxes.length) return;
        var now = Date.now();
        boxes.forEach(function(box) {
            var startMs = parseInt(box.getAttribute('data-start-ms'), 10);
            var timerEl = box.querySelector('.reasoning-timer');
            if (startMs && timerEl) {
                var sec = Math.max(0.1, (now - startMs) / 1000);
                timerEl.textContent = sec.toFixed(1) + 's';
            }
        });
    }, 100);

    // 4. Fixed Sidebar (Drag-resizing disabled; smooth CSS transition & auto-collapse on narrow window)
    var userCollapsedOnDesktop = false;
    var responsiveInitialized = false;

    function clearInlineSidebarWidth() {
        var sidebar = document.getElementById('sidebar');
        if (sidebar) {
            ['width', 'min-width', 'max-width', 'flex-basis', 'flex-grow', 'flex-shrink']
                .forEach(function(prop) { sidebar.style.removeProperty(prop); });
        }
        var existingResizer = document.getElementById('sidebar-resizer');
        if (existingResizer && existingResizer.parentNode) {
            existingResizer.parentNode.removeChild(existingResizer);
        }
    }

    var DESKTOP_PLACEHOLDER = 'Ask anything about your document corpus...';
    var MOBILE_PLACEHOLDER = 'Ask anything...';
    function applyMobilePlaceholder() {
        var ta = document.querySelector('#chat-input textarea');
        if (ta) ta.placeholder = NARROW_QUERY.matches ? MOBILE_PLACEHOLDER : DESKTOP_PLACEHOLDER;
    }

    function syncResponsiveSidebar() {
        var appShell = document.getElementById('app-shell');
        if (!appShell) return;
        if (NARROW_QUERY.matches) {
            appShell.classList.add('collapsed');
        } else if (!userCollapsedOnDesktop) {
            appShell.classList.remove('collapsed');
        }
    }

    document.addEventListener('click', function(e) {
        var collapseBtn = e.target && e.target.closest ? e.target.closest('#sidebar-collapse') : null;
        var expandBtn = e.target && e.target.closest ? e.target.closest('#sidebar-expand') : null;
        var sidebarAction = e.target && e.target.closest ? e.target.closest('#sidebar .sidebar-item, #new-chat-btn') : null;
        var insideSidebar = e.target && e.target.closest ? e.target.closest('#sidebar') : null;
        if (collapseBtn && !NARROW_QUERY.matches) {
            userCollapsedOnDesktop = true;
        } else if (expandBtn && !NARROW_QUERY.matches) {
            userCollapsedOnDesktop = false;
        } else if (NARROW_QUERY.matches) {
            var appShell = document.getElementById('app-shell');
            if (appShell && !appShell.classList.contains('collapsed')) {
                if (sidebarAction || (!insideSidebar && !expandBtn)) {
                    appShell.classList.add('collapsed');
                }
            }
        }
    }, true);

    if (NARROW_QUERY.addEventListener) {
        NARROW_QUERY.addEventListener('change', function() {
            applyMobilePlaceholder();
            clearInlineSidebarWidth();
            syncResponsiveSidebar();
        });
    }

    function initResizer() {
        var sidebar = document.getElementById('sidebar');
        if (!sidebar) return false;
        clearInlineSidebarWidth();
        if (!responsiveInitialized) {
            responsiveInitialized = true;
            syncResponsiveSidebar();
        }
        return true;
    }

    // 5. Gemini-style Plus Menu for Source Mode Selection
    function setupPlusMenu() {
        var plusBtn = document.getElementById('plus-btn');
        var dropdown = document.getElementById('plus-dropdown');
        var container = document.getElementById('plus-menu-container');
        if (!plusBtn || !dropdown || !container) return false;

        if (plusBtn.__menu_initialized__) return true;
        plusBtn.__menu_initialized__ = true;

        function setDropdownVisible(open) {
            if (open) {
                dropdown.classList.add('is-open');
                plusBtn.classList.add('is-open');
                plusBtn.setAttribute('aria-expanded', 'true');
            } else {
                dropdown.classList.remove('is-open');
                plusBtn.classList.remove('is-open');
                plusBtn.setAttribute('aria-expanded', 'false');
            }
        }

        plusBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            var isOpen = dropdown.classList.contains('is-open');
            setDropdownVisible(!isOpen);
        });

        document.addEventListener('click', function(e) {
            if (!container.contains(e.target)) {
                setDropdownVisible(false);
            }
        });

        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                setDropdownVisible(false);
            }
        });

        function updateChip(mode) {
            var chipContainer = document.getElementById('command-chip-container');
            var chipWrapper = document.getElementById('command-chip-wrapper');
            if (!chipContainer) return;

            if (mode && (mode === 'Assistant mode' || mode.indexOf('Assistant') !== -1 || mode.indexOf('All') !== -1 || mode.indexOf('Internet') !== -1)) {
                chipContainer.innerHTML = '<div class="cmd-chip"><span class="cmd-chip-text">Assistant mode</span><button type="button" class="cmd-chip-remove" title="Reset to RAG only" aria-label="Reset to RAG only">×</button></div>';
                chipContainer.style.display = 'block';
                if (chipWrapper) chipWrapper.style.setProperty('display', 'flex', 'important');
            } else {
                chipContainer.innerHTML = '';
                chipContainer.style.display = 'none';
                if (chipWrapper) chipWrapper.style.setProperty('display', 'none', 'important');
            }

            var removeBtn = chipContainer.querySelector('.cmd-chip-remove');
            if (removeBtn) {
                removeBtn.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    selectMode('RAG only');
                });
            }
        }

        function setGradioMode(val) {
            var box = document.getElementById('selected-source-mode');
            if (!box) return;
            var el = box.querySelector('textarea') || box.querySelector('input');
            if (el) {
                el.value = val;
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }

        function selectMode(targetMode) {
            targetMode = (targetMode === 'Assistant mode') ? 'Assistant mode' : 'RAG only';
            var items = dropdown.querySelectorAll('.plus-dropdown-item');
            items.forEach(function(item) {
                var m = item.getAttribute('data-mode');
                var isTarget = (m === targetMode);
                if (isTarget) {
                    item.classList.add('active');
                } else {
                    item.classList.remove('active');
                }
            });

            updateChip(targetMode);
            setGradioMode(targetMode);
            setDropdownVisible(false);

            // Refocus text input
            focusInput();
        }

        var items = dropdown.querySelectorAll('.plus-dropdown-item');
        items.forEach(function(item) {
            item.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                if (item.classList.contains('is-disabled') || item.getAttribute('aria-disabled') === 'true') {
                    return;
                }
                var mode = item.getAttribute('data-mode') || 'RAG only';
                selectMode(mode);
            });
        });

        // Initialize chip and mode on setup (Default: RAG only)
        updateChip('RAG only');

        window.resetSourceMode = function() {
            selectMode('RAG only');
        };

        return true;
    }

    // 6. Accessible names for controls gradio renders unnamed: the icon-only
    //    conversation delete button (re-created on every sidebar re-render),
    //    the send/stop circles, and the chat/search textareas ("Textbox")
    function nameControls() {
        document.querySelectorAll('.conv-delete:not([aria-label])').forEach(function(btn) {
            btn.setAttribute('aria-label', 'Delete conversation');
            btn.setAttribute('title', 'Delete conversation');
        });
        document.querySelectorAll('.sidebar-item:not([title]), .conv-title:not([title])').forEach(function(btn) {
            var txt = (btn.textContent || '').trim();
            if (txt) btn.setAttribute('title', txt);
        });
        var named = [
            ['#chat-input textarea', 'Message'],
            ['#conv-search textarea', 'Search conversations'],
            ['#submit-btn', 'Send message'],
            ['#stop-btn', 'Stop generating']
        ];
        named.forEach(function(pair) {
            var el = document.querySelector(pair[0]);
            if (el && !el.getAttribute('aria-label')) el.setAttribute('aria-label', pair[1]);
        });
    }

    var nameQueued = false;
    var namer = new MutationObserver(function() {
        if (nameQueued) return;
        nameQueued = true;
        requestAnimationFrame(function() {
            nameQueued = false;
            nameControls();
            if (!responsiveInitialized) initResizer();
            setupPlusMenu();
            applyMobilePlaceholder();
        });
    });

    function startNamer() {
        if (!document.body) { setTimeout(startNamer, 100); return; }
        namer.observe(document.body, { childList: true, subtree: true });
        nameControls();
        applyMobilePlaceholder();
    }
    startNamer();

    // 7. Real-time submit button state sync & dynamic input height (>= 2 lines -> > 2x height)
    var inputMirror = null;
    function getInputMirror() {
        if (inputMirror && document.body && document.body.contains(inputMirror)) {
            return inputMirror;
        }
        if (!document.body) return null;
        inputMirror = document.createElement('textarea');
        inputMirror.id = 'chat-input-mirror';
        inputMirror.setAttribute('aria-hidden', 'true');
        inputMirror.tabIndex = -1;
        inputMirror.style.cssText = [
            'position:fixed',
            'top:-9999px',
            'left:-9999px',
            'visibility:hidden',
            'pointer-events:none',
            'height:0px',
            'min-height:0px',
            'max-height:none',
            'padding:10px 8px',
            'border:none',
            'margin:0',
            'font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
            'font-size:13px',
            'font-weight:400',
            'font-style:normal',
            'line-height:20px',
            'box-sizing:border-box',
            'white-space:pre-wrap',
            'word-wrap:break-word',
            'overflow-wrap:break-word',
            'overflow-y:hidden'
        ].join(';');
        document.body.appendChild(inputMirror);
        return inputMirror;
    }

    function applyMultilineState(isMulti, ta, container) {
        var row = container ? container.querySelector('.input-row') : null;
        if (isMulti) {
            if (container) container.classList.add('is-multiline');
            if (document.body) document.body.classList.add('chat-input-multiline');
            if (row) {
                row.style.setProperty('min-height', '116px', 'important');
                row.style.setProperty('height', 'auto', 'important');
                row.style.setProperty('border-radius', '20px', 'important');
                row.style.setProperty('align-items', 'flex-end', 'important');
            }
            if (ta) {
                ta.style.setProperty('height', '104px', 'important');
                ta.style.setProperty('min-height', '104px', 'important');
                ta.style.setProperty('max-height', '180px', 'important');
                ta.style.setProperty('overflow-y', 'auto', 'important');
            }
        } else {
            if (container) container.classList.remove('is-multiline');
            if (document.body) document.body.classList.remove('chat-input-multiline');
            if (row) {
                row.style.setProperty('min-height', '48px', 'important');
                row.style.setProperty('height', '48px', 'important');
                row.style.setProperty('border-radius', '26px', 'important');
                row.style.setProperty('align-items', 'center', 'important');
            }
            if (ta) {
                ta.style.setProperty('height', '40px', 'important');
                ta.style.setProperty('min-height', '40px', 'important');
                ta.style.setProperty('max-height', '40px', 'important');
                ta.style.setProperty('overflow-y', 'hidden', 'important');
            }
        }
    }

    function syncInputHeight(explicitText, forceMultiline) {
        var ta = document.querySelector('#chat-input textarea');
        var container = document.getElementById('custom-input-container');
        if (!ta || !container) return;

        var val = (typeof explicitText === 'string') ? explicitText : (ta.value || '');
        if (!val) {
            applyMultilineState(false, ta, container);
            return;
        }

        var lineBreaks = (val.match(/\r?\n/g) || []).length;
        var isMulti = Boolean(forceMultiline) || (lineBreaks >= 1);

        if (!isMulti) {
            var w = ta.clientWidth || ta.offsetWidth || 620;
            var charsPerLine = Math.max(40, Math.floor((w - 20) / 7.2));
            if (val.length > charsPerLine) {
                isMulti = true;
            } else {
                var mirror = getInputMirror();
                if (mirror) {
                    mirror.style.width = w + 'px';
                    mirror.value = val;
                    // Single line is 10px top + 20px line-height + 10px bottom = 40px.
                    // 2 or more wrapped lines is >= 60px.
                    if (mirror.scrollHeight >= 50) {
                        isMulti = true;
                    }
                }
            }
        }

        applyMultilineState(isMulti, ta, container);
    }

    function syncSubmitBtn(explicitText, forceMultiline) {
        var ta = document.querySelector('#chat-input textarea');
        var btn = document.getElementById('submit-btn');
        var val = (typeof explicitText === 'string') ? explicitText : (ta ? ta.value : '');
        if (btn) {
            var hasText = Boolean(val && val.trim().length > 0);
            btn.style.opacity = hasText ? '1' : '0.5';
            btn.style.pointerEvents = hasText ? 'auto' : 'none';
        }
        syncInputHeight(val, forceMultiline);
    }

    window.syncChatInputUI = syncSubmitBtn;

    document.addEventListener('input', function(e) {
        if (e.target && e.target.matches('#chat-input textarea')) {
            syncSubmitBtn(e.target.value, false);
        }
    }, true);

    document.addEventListener('keyup', function(e) {
        if (e.target && e.target.matches('#chat-input textarea')) {
            syncSubmitBtn(e.target.value, false);
        }
    }, true);

    document.addEventListener('paste', function(e) {
        var ta = (e.target && e.target.matches('#chat-input textarea'))
            ? e.target
            : (e.target && e.target.closest ? e.target.closest('#chat-input') && document.querySelector('#chat-input textarea') : null);
        if (ta) {
            var pasted = '';
            try {
                pasted = (e.clipboardData || window.clipboardData).getData('text') || '';
            } catch (err) {}
            var pastedHasNewline = /\r?\n/.test(pasted);
            setTimeout(function() {
                syncSubmitBtn(ta.value, pastedHasNewline);
            }, 0);
            setTimeout(function() {
                syncSubmitBtn(ta.value, pastedHasNewline);
            }, 50);
        }
    }, true);

    document.addEventListener('keydown', function(e) {
        if (e.target && e.target.matches('#chat-input textarea') && e.key === 'Enter' && !e.shiftKey) {
            setTimeout(function() {
                syncSubmitBtn('', false);
            }, 50);
        }
    }, true);

    document.addEventListener('click', function(e) {
        if (e.target && (e.target.id === 'submit-btn' || e.target.closest('#submit-btn') || e.target.closest('.sidebar-item'))) {
            setTimeout(function() {
                syncSubmitBtn();
            }, 60);
            setTimeout(function() {
                syncSubmitBtn();
            }, 180);
        }
    }, true);

    // Initialize once Gradio renders #sidebar & #plus-menu-container
    var attempts = 0;
    var timer = setInterval(function() {
        attempts++;
        var r1 = initResizer();
        var r2 = setupPlusMenu();
        syncSubmitBtn();
        if ((r1 && r2) || attempts >= 30) {
            clearInterval(timer);
        }
    }, 100);
})();
