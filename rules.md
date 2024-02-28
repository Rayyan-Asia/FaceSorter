# Development Rules

## Notebook Usage Guidelines

1. Each person wanting to make tests should have their own notebook files. Do not edit a notebook file that wasn't yours without permission. This will help to minimize conflicts and maintain clarity in the development process.

2. If you want to work on something similar to an existing notebook, it's acceptable to create a new one. However, ensure that the new notebook is named in the following format: `{developer-name]-{0-9]*` to indicate ownership and versioning.

## Dependency Management

1. When adding a new tool or library (such as pandas or numpy), ensure to add it to the `requirements.txt` file. This allows other team members to easily install the dependencies with a simple command and continue working seamlessly.

2. It's important to keep the `requirements.txt` file updated with any new dependencies introduced during the development process. This ensures that all team members are using the same environment and have access to the necessary tools.
