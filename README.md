# DoTemplate (0.0.0)

A template for python projects.

## Using the Init Script

The `init` script is used to initialize a new project from this template. It automatically replaces template placeholders with your project-specific information and sets up a new git repository.

### Prerequisites

Before running the init script, make sure you have:

- Git configured with your user name and email
- `uv` package manager installed
- `just` command runner installed

### Basic Usage

```bash
./init my-project-name
```

This will create a new project with the name "my-project-name" and default settings.

### Command Line Options

| Option                  | Short | Default          | Description                               |
| ----------------------- | ----- | ---------------- | ----------------------------------------- |
| `name`                  | -     | `dotemplate`     | Name of the new project                   |
| `--capitalized-name`    | `-N`  | Auto-generated   | Capitalized version of the project name   |
| `--description`         | `-d`  | Empty            | Project description                       |
| `--github-organization` | `-o`  | `doplaydo`       | GitHub organization for the repository    |
| `--version`             | `-v`  | `0.0.0`          | Initial version number                    |
| `--initial-commit`      | `-c`  | `Initial Commit` | Message for the initial git commit        |
| `--no-delete-init`      | `-D`  | False            | Keep the init script after initialization |

### Examples

**Basic project initialization:**

```bash
./init my-awesome-project
```

**Full customization:**

```bash
./init my-project \
  --capitalized-name "MyProject" \
  --description "An awesome Python project" \
  --github-organization "myorg" \
  --version "1.0.0" \
  --initial-commit "🎉 Initial release"
```

**Keep the init script for reference:**

```bash
./init my-project --no-delete-init
```

### What the Script Does

1. **Validates environment**: Checks for git configuration and required tools
2. **Cleans repository**: Runs `just clean` to remove build artifacts
3. **Makes replacements**: Updates all files in the repository, replacing:
   - `dotemplate` → your project name
   - `DoTemplate` → your capitalized project name
   - `0.0.0` → your specified version
   - `doplaydo` → your GitHub organization
4. **Updates configuration files**:
   - Sets author information in `pyproject.toml`
   - Updates description in `pyproject.toml` and `mkdocs.yml`
5. **Initializes git repository**: Creates a fresh git repo with your initial commit
6. **Sets up development tools**: Installs pre-commit hooks
7. **Cleans up**: Removes the init script (unless `--no-delete-init` is used)

### After Initialization

After running the init script, you'll have a fully configured Python project ready for development. The repository will be initialized with git, and all template placeholders will be replaced with your project-specific information.
