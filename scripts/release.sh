#!/bin/bash
#
# PanelBox Release Script
# Usage: ./scripts/release.sh [test|prod]
#
# This script automates the release process for PanelBox
# Use 'test' to upload to Test PyPI, 'prod' to upload to production PyPI

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "Must run from project root directory"
    exit 1
fi

# Get version from pyproject.toml
VERSION=$(grep '^version = ' pyproject.toml | cut -d '"' -f 2)
print_step "Preparing release for version ${VERSION}"

# Determine target
TARGET=${1:-test}
if [ "$TARGET" != "test" ] && [ "$TARGET" != "prod" ]; then
    print_error "Invalid target. Use 'test' or 'prod'"
    exit 1
fi

# Step 1: Run tests
print_step "Running tests..."
python3 -m pytest tests/validation/robustness/ -v --tb=short > /tmp/test_output.txt 2>&1
if [ $? -eq 0 ]; then
    PASSED=$(grep -o '[0-9]* passed' /tmp/test_output.txt | cut -d ' ' -f 1)
    SKIPPED=$(grep -o '[0-9]* skipped' /tmp/test_output.txt | cut -d ' ' -f 1 || echo "0")
    print_success "Tests passed: ${PASSED} passed, ${SKIPPED} skipped"
else
    print_error "Tests failed! Check /tmp/test_output.txt"
    exit 1
fi

# Step 2: Verify version
print_step "Verifying version..."
PYTHON_VERSION=$(python3 -c "import panelbox; print(panelbox.__version__)")
if [ "$PYTHON_VERSION" != "$VERSION" ]; then
    print_error "Version mismatch! pyproject.toml: ${VERSION}, __version__.py: ${PYTHON_VERSION}"
    exit 1
fi
print_success "Version verified: ${VERSION}"

# Step 3: Clean previous builds
print_step "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info
print_success "Clean complete"

# Step 4: Build package
print_step "Building package..."
python3 -m build > /tmp/build_output.txt 2>&1
if [ $? -eq 0 ]; then
    print_success "Build complete"
else
    print_error "Build failed! Check /tmp/build_output.txt"
    exit 1
fi

# Step 5: Check package
print_step "Checking package..."
twine check dist/* > /tmp/twine_check.txt 2>&1
if [ $? -eq 0 ]; then
    print_success "Package check passed"
else
    print_error "Package check failed! Check /tmp/twine_check.txt"
    exit 1
fi

# Step 6: List built files
print_step "Built files:"
ls -lh dist/
echo ""

# Step 7: Upload
if [ "$TARGET" == "test" ]; then
    print_step "Uploading to Test PyPI..."
    print_warning "You will need Test PyPI credentials"
    twine upload --repository testpypi dist/*
    if [ $? -eq 0 ]; then
        print_success "Upload to Test PyPI complete"
        echo ""
        echo -e "${GREEN}✓ Success!${NC} View at: https://test.pypi.org/project/panelbox/${VERSION}/"
        echo ""
        echo "To test installation:"
        echo "  pip install --index-url https://test.pypi.org/simple/ panelbox==${VERSION}"
    else
        print_error "Upload to Test PyPI failed"
        exit 1
    fi
else
    print_warning "You are about to upload to PRODUCTION PyPI"
    read -p "Are you sure? (yes/no): " -r
    echo
    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        print_step "Uploading to PyPI..."
        twine upload dist/*
        if [ $? -eq 0 ]; then
            print_success "Upload to PyPI complete"
            echo ""
            echo -e "${GREEN}🎉 SUCCESS! v${VERSION} is now live on PyPI!${NC}"
            echo ""
            echo "View at: https://pypi.org/project/panelbox/${VERSION}/"
            echo ""
            echo "Users can install with:"
            echo "  pip install --upgrade panelbox"
            echo ""
            echo "Don't forget to:"
            echo "  1. Create GitHub release"
            echo "  2. Update documentation"
            echo "  3. Announce release"
        else
            print_error "Upload to PyPI failed"
            exit 1
        fi
    else
        print_warning "Upload cancelled"
        exit 0
    fi
fi

echo ""
print_success "Release process complete!"
