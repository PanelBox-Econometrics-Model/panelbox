/**
 * PanelBox Reports - Tab Navigation
 * ==================================
 *
 * Handles tab switching for interactive reports.
 * Lightweight vanilla JavaScript implementation.
 *
 * Version: 1.0.0
 */

(function() {
    'use strict';

    /**
     * Initialize tab navigation for all tab containers
     */
    function initTabs() {
        const tabContainers = document.querySelectorAll('.tabs-container');

        tabContainers.forEach(container => {
            const tabButtons = container.querySelectorAll('.tab-button');
            const tabContents = container.querySelectorAll('.tab-content');

            // Set up click handlers
            tabButtons.forEach(button => {
                button.addEventListener('click', function() {
                    const targetId = this.dataset.tab;
                    switchTab(container, targetId);
                });
            });

            // Activate first tab by default
            if (tabButtons.length > 0) {
                const firstTab = tabButtons[0].dataset.tab;
                switchTab(container, firstTab);
            }
        });
    }

    /**
     * Switch to a specific tab
     * @param {HTMLElement} container - The tabs container element
     * @param {string} targetId - The ID of the tab to activate
     */
    function switchTab(container, targetId) {
        const tabButtons = container.querySelectorAll('.tab-button');
        const tabContents = container.querySelectorAll('.tab-content');

        // Deactivate all tabs
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(content => content.classList.remove('active'));

        // Activate target tab
        const targetButton = container.querySelector(`[data-tab="${targetId}"]`);
        const targetContent = document.getElementById(targetId);

        if (targetButton && targetContent) {
            targetButton.classList.add('active');
            targetContent.classList.add('active');

            // Scroll tab into view if needed
            targetButton.scrollIntoView({
                behavior: 'smooth',
                block: 'nearest',
                inline: 'center'
            });

            // Dispatch custom event for other components
            const event = new CustomEvent('tabChanged', {
                detail: { tabId: targetId }
            });
            container.dispatchEvent(event);
        }
    }

    /**
     * Add keyboard navigation support
     */
    function initKeyboardNavigation() {
        document.addEventListener('keydown', function(e) {
            const activeTab = document.querySelector('.tab-button.active');

            if (!activeTab) return;

            const container = activeTab.closest('.tabs-container');
            const tabButtons = Array.from(container.querySelectorAll('.tab-button'));
            const currentIndex = tabButtons.indexOf(activeTab);

            let nextIndex;

            // Arrow key navigation
            if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                e.preventDefault();
                nextIndex = currentIndex > 0 ? currentIndex - 1 : tabButtons.length - 1;
            } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                e.preventDefault();
                nextIndex = currentIndex < tabButtons.length - 1 ? currentIndex + 1 : 0;
            } else if (e.key === 'Home') {
                e.preventDefault();
                nextIndex = 0;
            } else if (e.key === 'End') {
                e.preventDefault();
                nextIndex = tabButtons.length - 1;
            }

            if (nextIndex !== undefined) {
                const nextTab = tabButtons[nextIndex];
                const targetId = nextTab.dataset.tab;
                switchTab(container, targetId);
                nextTab.focus();
            }
        });
    }

    /**
     * Save/restore tab state in URL hash
     */
    function initHashNavigation() {
        // Restore tab from hash on load
        if (window.location.hash) {
            const tabId = window.location.hash.substring(1);
            const tabButton = document.querySelector(`[data-tab="${tabId}"]`);
            if (tabButton) {
                const container = tabButton.closest('.tabs-container');
                switchTab(container, tabId);
            }
        }

        // Update hash when tab changes
        document.addEventListener('tabChanged', function(e) {
            const tabId = e.detail.tabId;
            if (history.replaceState) {
                history.replaceState(null, null, '#' + tabId);
            } else {
                window.location.hash = tabId;
            }
        });
    }

    /**
     * Initialize everything when DOM is ready
     */
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            initTabs();
            initKeyboardNavigation();
            initHashNavigation();
        });
    } else {
        initTabs();
        initKeyboardNavigation();
        initHashNavigation();
    }

    // Export functions for external use
    window.PanelBoxTabs = {
        init: initTabs,
        switchTab: switchTab
    };

})();
