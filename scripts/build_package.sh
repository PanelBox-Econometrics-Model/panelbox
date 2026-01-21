#!/bin/bash
# Build script for PanelBox package
# Usage: bash scripts/build_package.sh

set -e  # Exit on error

echo "========================================"
echo "PanelBox Package Build Script"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if required tools are installed
echo "1. Checking build tools..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python ${PYTHON_VERSION}${NC}"

# Check for build module
if python3 -c "import build" 2>/dev/null; then
    echo -e "${GREEN}✓ build module installed${NC}"
else
    echo -e "${YELLOW}⚠ build module not installed${NC}"
    echo "Installing build module..."
    pip install build
fi

# Check for twine
if command -v twine &> /dev/null; then
    echo -e "${GREEN}✓ twine installed${NC}"
else
    echo -e "${YELLOW}⚠ twine not installed (needed for checking distribution)${NC}"
    echo "Install with: pip install twine"
fi
echo ""

# Clean previous builds
echo "2. Cleaning previous builds..."
if [ -d "build" ]; then
    rm -rf build/
    echo -e "${GREEN}✓ Removed build/${NC}"
fi

if [ -d "dist" ]; then
    rm -rf dist/
    echo -e "${GREEN}✓ Removed dist/${NC}"
fi

find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
echo -e "${GREEN}✓ Removed .egg-info${NC}"

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo -e "${GREEN}✓ Removed Python cache${NC}"
echo ""

# Check version
echo "3. Checking version..."
VERSION=$(python3 -c "import sys; sys.path.insert(0, 'panelbox'); from __version__ import __version__; print(__version__)")
echo -e "${BLUE}Version: ${VERSION}${NC}"
echo ""

# Build package
echo "4. Building package..."
if python3 -m build; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
echo ""

# List built files
echo "5. Distribution files created:"
ls -lh dist/
echo ""

# Check package with twine
if command -v twine &> /dev/null; then
    echo "6. Checking package with twine..."
    if twine check dist/*; then
        echo -e "${GREEN}✓ Package check passed${NC}"
    else
        echo -e "${RED}✗ Package check failed${NC}"
        exit 1
    fi
else
    echo "6. Skipping twine check (not installed)"
fi
echo ""

# Show package contents summary
echo "7. Package contents summary:"
echo ""
echo "Wheel (.whl) contents:"
unzip -l dist/*.whl | grep "panelbox/" | head -10
echo "  ... (showing first 10 files)"
echo ""

echo "Source distribution (.tar.gz) contents:"
tar -tzf dist/*.tar.gz | head -10
echo "  ... (showing first 10 files)"
echo ""

# Summary
echo "========================================"
echo "Build Complete!"
echo "========================================"
echo ""
echo -e "${GREEN}✓ Package built successfully${NC}"
echo -e "Version: ${BLUE}${VERSION}${NC}"
echo ""
echo "Files created:"
echo "  - dist/panelbox-${VERSION}.tar.gz"
echo "  - dist/panelbox-${VERSION}-py3-none-any.whl"
echo ""
echo "Next steps:"
echo "  1. Test installation locally:"
echo "     pip install dist/panelbox-${VERSION}-py3-none-any.whl"
echo ""
echo "  2. Upload to Test PyPI:"
echo "     twine upload --repository testpypi dist/*"
echo ""
echo "  3. Upload to PyPI (production):"
echo "     twine upload dist/*"
echo ""
echo "See PUBLISHING_GUIDE.md for detailed instructions."
