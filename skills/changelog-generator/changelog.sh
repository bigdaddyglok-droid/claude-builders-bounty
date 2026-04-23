#!/bin/bash

# CHANGELOG Generator - Generates structured CHANGELOG.md from git history
# Usage: bash changelog.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}Error: Not a git repository${NC}"
    exit 1
fi

echo -e "${YELLOW}Generating CHANGELOG.md...${NC}"

# Create backup if CHANGELOG.md exists
if [ -f CHANGELOG.md ]; then
    cp CHANGELOG.md CHANGELOG.md.bak
    echo -e "${YELLOW}Backed up existing CHANGELOG.md to CHANGELOG.md.bak${NC}"
fi

# Function to process commits and categorize them
process_commits() {
    local RANGE=$1
    
    # Get commits in range
    local COMMITS=$(git log $RANGE --pretty=format:"%h %s" 2>/dev/null || echo "")
    
    if [ -z "$COMMITS" ]; then
        echo "No changes"
        return
    fi
    
    # Arrays to store categorized commits
    local ADDED=()
    local FIXED=()
    local CHANGED=()
    local REMOVED=()
    
    # Categorize commits
    while IFS= read -r COMMIT; do
        if [[ $COMMIT =~ ^[a-f0-9]+\ feat ]]; then
            ADDED+=("- ${COMMIT#* }")
        elif [[ $COMMIT =~ ^[a-f0-9]+\ fix ]]; then
            FIXED+=("- ${COMMIT#* }")
        elif [[ $COMMIT =~ ^[a-f0-9]+\ (refactor|docs|style|perf) ]]; then
            CHANGED+=("- ${COMMIT#* }")
        elif [[ $COMMIT =~ ^[a-f0-9]+\ (remove|drop) ]]; then
            REMOVED+=("- ${COMMIT#* }")
        fi
    done <<< "$COMMITS"
    
    # Output categorized commits
    if [ ${#ADDED[@]} -gt 0 ]; then
        echo "### Added"
        echo ""
        printf '%s\n' "${ADDED[@]}"
        echo ""
    fi
    
    if [ ${#FIXED[@]} -gt 0 ]; then
        echo "### Fixed"
        echo ""
        printf '%s\n' "${FIXED[@]}"
        echo ""
    fi
    
    if [ ${#CHANGED[@]} -gt 0 ]; then
        echo "### Changed"
        echo ""
        printf '%s\n' "${CHANGED[@]}"
        echo ""
    fi
    
    if [ ${#REMOVED[@]} -gt 0 ]; then
        echo "### Removed"
        echo ""
        printf '%s\n' "${REMOVED[@]}"
        echo ""
    fi
}

# Initialize CHANGELOG
cat > CHANGELOG.md << 'EOF'
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

EOF

# Get all tags sorted by version
TAGS=$(git tag --sort=-version:refname 2>/dev/null || git tag -l | sort -rV)

if [ -z "$TAGS" ]; then
    # No tags, process all commits as "Unreleased"
    echo "## [Unreleased]" >> CHANGELOG.md
    echo "" >> CHANGELOG.md
    
    process_commits "HEAD" >> CHANGELOG.md
else
    # Process each tag
    PREV_TAG=""
    for TAG in $TAGS; do
        if [ -z "$PREV_TAG" ]; then
            # Latest tag
            COMMIT_RANGE="$TAG"
        else
            # Between tags
            COMMIT_RANGE="$TAG..$PREV_TAG"
        fi
        
        # Get tag date
        TAG_DATE=$(git log -1 --format=%ai "$TAG" | cut -d' ' -f1)
        
        echo "## [$TAG] - $TAG_DATE" >> CHANGELOG.md
        echo "" >> CHANGELOG.md
        
        process_commits "$COMMIT_RANGE" >> CHANGELOG.md
        
        PREV_TAG="$TAG"
    done
    
    # Process commits since last tag
    if [ -n "$PREV_TAG" ]; then
        echo "## [Unreleased]" >> CHANGELOG.md
        echo "" >> CHANGELOG.md
        process_commits "$PREV_TAG..HEAD" >> CHANGELOG.md
    fi
fi

echo -e "${GREEN}✓ CHANGELOG.md generated successfully${NC}"
echo -e "${GREEN}Location: $(pwd)/CHANGELOG.md${NC}"
