# Contributing to Firco XGBoost Compliance Predictor

Thank you for your interest in contributing to the Firco XGBoost Compliance Predictor! We welcome contributions from the community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)

## 📜 Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## 🤝 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, dependencies)
- **Logs or screenshots** if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- **Clear use case** for the enhancement
- **Expected behavior** and benefits
- **Possible implementation** approach (if known)

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch** from `main`
3. **Make your changes**
4. **Test thoroughly**
5. **Submit a pull request**

## 🛠️ Development Setup

### Prerequisites

- Python 3.9 or higher
- MongoDB (local or Atlas)
- Git

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/XG-Boast-TFIDF.git
cd XG-Boast-TFIDF/Firco/xgb

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your MongoDB credentials
nano .env

# Run the application
uvicorn xgb_app_F:app --reload
```

## 📝 Coding Standards

### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **4 spaces** for indentation (no tabs)
- Maximum line length: **120 characters**
- Use **snake_case** for functions and variables
- Use **PascalCase** for classes
- Add **docstrings** to all public functions and classes

### Docstring Format

```python
def train_model(csv_file: str, test_size: float = 0.2) -> dict:
    """
    Train XGBoost model on compliance data.
    
    Args:
        csv_file: Path to the training CSV file
        test_size: Proportion of data to use for testing (default: 0.2)
    
    Returns:
        Dictionary containing trained model and performance metrics
    
    Raises:
        FileNotFoundError: If csv_file does not exist
        ValueError: If test_size is not between 0 and 1
    """
    pass
```

### Type Hints

Use type hints for function parameters and return values:

```python
from typing import Dict, List, Optional

def predict(model: Any, data: Dict[str, Any]) -> List[str]:
    """Make predictions using the trained model."""
    pass
```

### Imports

Organize imports in this order:

```python
# Standard library imports
import os
import sys
from typing import Dict, List

# Third-party imports
import pandas as pd
import numpy as np
from fastapi import FastAPI

# Local imports
from config import MODEL_SAVE_DIR
from models import FircoHierarchicalXGBoost
```

## 📏 Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, no logic change)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```bash
# Good commits
feat(api): add endpoint for batch predictions
fix(model): correct feature engineering for MT202 messages
docs(readme): update installation instructions
refactor(utils): simplify data preprocessing pipeline

# Bad commits
update stuff
fix bug
changes
WIP
```

### Commit Best Practices

- Keep commits **atomic** (one logical change per commit)
- Write **clear, descriptive** commit messages
- Reference **issue numbers** when applicable: `fixes #123`
- Commit **frequently** with meaningful progress

## 🔄 Pull Request Process

### Before Submitting

1. **Update documentation** if needed
2. **Add tests** for new features
3. **Ensure all tests pass**
4. **Update CHANGELOG** if applicable
5. **Rebase on latest main** branch

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] Added new tests for changes
- [ ] Tested locally

## Checklist
- [ ] Code follows style guidelines
- [ ] Added/updated documentation
- [ ] No new warnings
- [ ] Rebased on latest main
```

### Review Process

1. **Automated checks** must pass (if configured)
2. **Code review** by maintainer(s)
3. **Address feedback** and make changes
4. **Final approval** and merge

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python test_api_F.py

# Run specific test
python test_batch_prediction.py

# With pytest (if configured)
pytest tests/ -v
```

### Writing Tests

```python
import pytest
from fastapi.testclient import TestClient
from xgb_app_F import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint returns 200."""
    response = client.get("/v1/firco-xgb/health")
    assert response.status_code == 200
    assert "status" in response.json()
```

### Test Coverage

- Aim for **>80% code coverage**
- Test **edge cases** and error conditions
- Include **integration tests** for API endpoints
- Test **data validation** logic

## 🔍 Code Review Checklist

### For Contributors

- [ ] Code is self-documenting and clean
- [ ] All functions have docstrings
- [ ] Type hints are used
- [ ] No hardcoded credentials or secrets
- [ ] Error handling is appropriate
- [ ] Logging is adequate
- [ ] Performance is acceptable

### For Reviewers

- [ ] Code meets quality standards
- [ ] Tests are comprehensive
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Changes are backward compatible
- [ ] Performance impact is acceptable

## 📚 Additional Resources

- [Python PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Conventional Commits](https://www.conventionalcommits.org/)

## 💬 Questions?

If you have questions about contributing:

- Open a [GitHub Discussion](https://github.com/AnirbanDB/XG-Boast-TFIDF/discussions)
- Create an [Issue](https://github.com/AnirbanDB/XG-Boast-TFIDF/issues)

## 🙏 Recognition

Contributors will be acknowledged in:
- Project README
- Release notes
- Contributors page

Thank you for making this project better! 🎉
