/**
 * PanelBox Reports - Utility Functions
 * =====================================
 *
 * Common utility functions for reports.
 *
 * Version: 1.0.0
 */

(function() {
    'use strict';

    /**
     * Format a number with specified decimal places
     * @param {number} value - The number to format
     * @param {number} decimals - Number of decimal places
     * @returns {string} Formatted number
     */
    function formatNumber(value, decimals = 3) {
        if (value === null || value === undefined || isNaN(value)) {
            return 'N/A';
        }
        return value.toFixed(decimals);
    }

    /**
     * Format a p-value with scientific notation if small
     * @param {number} pvalue - The p-value to format
     * @returns {string} Formatted p-value
     */
    function formatPValue(pvalue) {
        if (pvalue === null || pvalue === undefined || isNaN(pvalue)) {
            return 'N/A';
        }
        if (pvalue < 0.001) {
            return pvalue.toExponential(2);
        }
        return pvalue.toFixed(4);
    }

    /**
     * Format a percentage
     * @param {number} value - The value to format as percentage
     * @param {number} decimals - Number of decimal places
     * @returns {string} Formatted percentage
     */
    function formatPercentage(value, decimals = 2) {
        if (value === null || value === undefined || isNaN(value)) {
            return 'N/A';
        }
        return (value * 100).toFixed(decimals) + '%';
    }

    /**
     * Add significance stars to a p-value
     * @param {number} pvalue - The p-value
     * @returns {string} Stars string
     */
    function significanceStars(pvalue) {
        if (pvalue < 0.001) return '***';
        if (pvalue < 0.01) return '**';
        if (pvalue < 0.05) return '*';
        if (pvalue < 0.1) return '.';
        return '';
    }

    /**
     * Copy text to clipboard
     * @param {string} text - Text to copy
     * @returns {Promise<void>}
     */
    async function copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            showNotification('Copied to clipboard!', 'success');
        } catch (err) {
            // Fallback for older browsers
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();
            try {
                document.execCommand('copy');
                showNotification('Copied to clipboard!', 'success');
            } catch (err2) {
                showNotification('Failed to copy', 'danger');
            }
            document.body.removeChild(textarea);
        }
    }

    /**
     * Show a temporary notification
     * @param {string} message - Message to show
     * @param {string} type - Type: success, warning, danger, info
     * @param {number} duration - Duration in ms
     */
    function showNotification(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 24px;
            border-radius: 8px;
            background: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            z-index: 9999;
            animation: slideInRight 0.3s ease;
        `;

        // Set color based on type
        const colors = {
            success: '#10b981',
            warning: '#f59e0b',
            danger: '#ef4444',
            info: '#3b82f6'
        };
        notification.style.borderLeft = `4px solid ${colors[type] || colors.info}`;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, duration);
    }

    /**
     * Export table to CSV
     * @param {string} tableId - ID of the table element
     * @param {string} filename - Name for the CSV file
     */
    function exportTableToCSV(tableId, filename = 'table.csv') {
        const table = document.getElementById(tableId);
        if (!table) return;

        const rows = Array.from(table.querySelectorAll('tr'));
        const csv = rows.map(row => {
            const cells = Array.from(row.querySelectorAll('th, td'));
            return cells.map(cell => {
                let text = cell.textContent.trim();
                // Escape quotes and wrap in quotes if contains comma
                if (text.includes(',') || text.includes('"')) {
                    text = '"' + text.replace(/"/g, '""') + '"';
                }
                return text;
            }).join(',');
        }).join('\n');

        // Download CSV
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        showNotification('Table exported to CSV', 'success');
    }

    /**
     * Print the report
     */
    function printReport() {
        window.print();
    }

    /**
     * Toggle element visibility
     * @param {string} elementId - ID of element to toggle
     */
    function toggleVisibility(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = element.style.display === 'none' ? '' : 'none';
        }
    }

    /**
     * Smooth scroll to element
     * @param {string} elementId - ID of element to scroll to
     */
    function scrollToElement(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    /**
     * Debounce function
     * @param {Function} func - Function to debounce
     * @param {number} wait - Wait time in ms
     * @returns {Function} Debounced function
     */
    function debounce(func, wait = 300) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Check if element is in viewport
     * @param {HTMLElement} element - Element to check
     * @returns {boolean} True if in viewport
     */
    function isInViewport(element) {
        const rect = element.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    }

    // Add CSS for animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInRight {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        @keyframes slideOutRight {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);

    // Export utilities
    window.PanelBoxUtils = {
        formatNumber,
        formatPValue,
        formatPercentage,
        significanceStars,
        copyToClipboard,
        showNotification,
        exportTableToCSV,
        printReport,
        toggleVisibility,
        scrollToElement,
        debounce,
        isInViewport
    };

})();
