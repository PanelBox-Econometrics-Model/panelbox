#!/bin/bash
# Quality Assurance Script for PanelBox
# Runs code formatting, linting, and type checking

set -e  # Exit on error

echo "========================================"
echo "PanelBox Quality Assurance"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if in virtual environment (recommended)
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo -e "${YELLOW}⚠ Warning: Not in a virtual environment${NC}"
    echo ""
fi

# 1. Black - Code Formatting
echo "1. Running Black (code formatter)..."
if python3 -m black --check panelbox/ 2>/dev/null; then
    echo -e "${GREEN}✓ Black: Code is formatted correctly${NC}"
else
    echo -e "${YELLOW}⚠ Black: Code needs formatting${NC}"
    echo "  Run: python3 -m black panelbox/"
fi
echo ""

# 2. isort - Import Sorting
echo "2. Running isort (import sorter)..."
if python3 -m isort --check-only panelbox/ 2>/dev/null; then
    echo -e "${GREEN}✓ isort: Imports are sorted correctly${NC}"
else
    echo -e "${YELLOW}⚠ isort: Imports need sorting${NC}"
    echo "  Run: python3 -m isort panelbox/"
fi
echo ""

# 3. Flake8 - Linting
echo "3. Running Flake8 (linter)..."
if python3 -m flake8 panelbox/ 2>/dev/null; then
    echo -e "${GREEN}✓ Flake8: No linting issues${NC}"
else
    echo -e "${RED}✗ Flake8: Found linting issues${NC}"
    python3 -m flake8 panelbox/ | head -20
    echo "  (showing first 20 issues)"
fi
echo ""

# 4. MyPy - Type Checking (optional, often has many warnings)
echo "4. Running MyPy (type checker) - optional..."
if command -v mypy &> /dev/null; then
    # Just count errors, don't fail
    MYPY_ERRORS=$(python3 -m mypy panelbox/ 2>&1 | grep -c "error:" || true)
    if [ "$MYPY_ERRORS" -eq 0 ]; then
        echo -e "${GREEN}✓ MyPy: No type errors${NC}"
    else
        echo -e "${YELLOW}⚠ MyPy: Found $MYPY_ERRORS type issues${NC}"
        echo "  (Type hints are optional for release)"
    fi
else
    echo -e "${YELLOW}⚠ MyPy not installed (optional)${NC}"
fi
echo ""

# Summary
echo "========================================"
echo "QA Check Complete"
echo "========================================"
echo ""
echo "To fix formatting issues automatically:"
echo "  python3 -m black panelbox/"
echo "  python3 -m isort panelbox/"
echo ""
echo "To install QA tools:"
echo "  pip install black isort flake8 mypy"
