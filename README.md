# spice.cloud agent

# Setup

Follow all getting started guides within the Wiki

```bash
git clone git@github.com:spicecloud/agent.git
cd agent
make setup
```

# Linting and Autoformatting

Linting:
`ruff`: https://github.com/charliermarsh/ruff

Formatting:
`black`: https://github.com/psf/black

## Troubleshooting

### macOS

#### ModuleNotFoundError: No module named '\_lzma'

WARNING: The Python lzma extension was not compiled. Missing the lzma lib?

```bash
brew install xz
```
