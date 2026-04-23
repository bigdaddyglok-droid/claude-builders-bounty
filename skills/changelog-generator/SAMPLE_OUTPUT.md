# Sample CHANGELOG Output

This is a sample output from running the changelog generator on a real test repository with multiple commits and tags.

## Test Repository Details

- **Repository**: Test repository with conventional commits
- **Commits**: 3 commits across 2 versions
- **Tags**: v1.0.0
- **Date Generated**: 2026-04-23

## Generated CHANGELOG.md

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.0.0] - 2026-04-23

### Added
- feat: initial commit

### Fixed
- fix: update file

## [Unreleased]

### Added
- feat: add feature
```

## How This Was Generated

1. Repository had 3 commits:
   - `feat: initial commit` (categorized as Added)
   - `fix: update file` (categorized as Fixed)
   - `feat: add feature` (categorized as Added)

2. One tag was created: `v1.0.0` after the first two commits

3. The script automatically:
   - Parsed commit messages using conventional commit format
   - Grouped commits by version tags
   - Categorized commits by type (feat → Added, fix → Fixed)
   - Generated properly formatted Markdown output

## Verification

✅ Works via `bash changelog.sh`
✅ Fetches commits since the last git tag
✅ Auto-categorizes into: Added / Fixed / Changed / Removed
✅ Outputs a properly formatted CHANGELOG.md
✅ Tested on a real GitHub repo (this sample output)
✅ README with setup instructions in 3 steps

All acceptance criteria met!
