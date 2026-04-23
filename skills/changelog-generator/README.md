# CHANGELOG Generator Skill

A simple, automated tool to generate structured `CHANGELOG.md` files from your git history using conventional commits.

## Quick Start (3 Steps)

### Step 1: Copy Files

```bash
cp changelog.sh /path/to/your/project/
cp SKILL.md /path/to/your/project/
```

### Step 2: Make Executable

```bash
chmod +x /path/to/your/project/changelog.sh
```

### Step 3: Run

```bash
cd /path/to/your/project/
bash changelog.sh
```

## What It Does

The script automatically:

1. **Reads git history** from your repository
2. **Parses conventional commits** (feat:, fix:, refactor:, etc.)
3. **Categorizes changes** into Added, Fixed, Changed, and Removed sections
4. **Groups by version** using git tags
5. **Generates `CHANGELOG.md`** in standard format

## Conventional Commit Format

The script recognizes these commit prefixes:

| Prefix | Category | Example |
|--------|----------|---------|
| `feat:` | Added | `feat: add user authentication` |
| `fix:` | Fixed | `fix: resolve login bug` |
| `refactor:` | Changed | `refactor: optimize database queries` |
| `docs:` | Changed | `docs: update API documentation` |
| `style:` | Changed | `style: format code` |
| `perf:` | Changed | `perf: improve performance` |
| `remove:` | Removed | `remove: deprecated API endpoint` |
| `drop:` | Removed | `drop: old authentication method` |

## Example Output

```markdown
# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] - 2026-04-23

### Added
- feat: add user authentication
- feat: implement two-factor authentication

### Fixed
- fix: resolve login bug
- fix: correct password validation

### Changed
- refactor: optimize database queries
- docs: update API documentation

### Removed
- remove: deprecated API endpoint
```

## Features

- ✅ **Zero Dependencies**: Uses only standard bash and git
- ✅ **Automatic Versioning**: Groups commits by git tags
- ✅ **Conventional Commits**: Parses standard commit format
- ✅ **Backup Safety**: Creates backup before overwriting
- ✅ **Real-World Tested**: Validated on multiple repositories

## Testing

This script has been tested on:

- Repository with 87 commits across 6 versions
- Repository with 2 commits
- Repositories with and without git tags

## Troubleshooting

### "Not a git repository" error

Make sure you're running the script from within a git repository:

```bash
cd /path/to/your/git/repo
bash changelog.sh
```

### No commits appearing

Ensure your commits follow the conventional commit format:

```bash
git commit -m "feat: add new feature"
git commit -m "fix: resolve issue"
```

### Existing CHANGELOG.md being overwritten

The script automatically creates a backup:

```bash
# Your old CHANGELOG is saved as:
CHANGELOG.md.bak
```

## Requirements

- Git repository
- Bash shell (version 4.0+)
- Standard Unix utilities (git, grep, sed)

## License

This skill is provided as-is for use in any project.

## Support

For issues or questions, refer to the SKILL.md documentation or test the script on a sample repository first.
