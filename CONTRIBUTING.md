# Contributing to Vision Transformer Tutorial

Thank you for your interest in contributing! This project aims to provide a clean, educational implementation of Vision Transformers for semantic segmentation.

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/vis-tutorial.git
cd vis-tutorial
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

```bash
bash data/setup_dataset.sh
```

### 5. Verify Setup

```bash
python examples/sanity_check.py
```

## Running Tests

Run the test suite to ensure everything works:

```bash
python -m pytest tests/ -v
```

Run specific test files:

```bash
python -m pytest tests/test_encoder.py -v
python -m pytest tests/test_decoder.py -v
```

## Code Style Guidelines

This project follows clean code principles:

### File Organization
- Files should be short (under 200 lines, ideally under 100)
- One class or concept per file
- Clear separation of concerns

### Naming Conventions
- Use meaningful names that reveal intent
- Functions: `verb_noun` (e.g., `create_model`, `compute_loss`)
- Classes: `PascalCase` (e.g., `PatchEmbedding`, `Segmenter`)
- Variables: `snake_case` (e.g., `batch_size`, `learning_rate`)

### Functions
- Each function should do one thing
- Keep functions short and focused
- Use early returns instead of nested if/else
- No side effects

### Comments
- Avoid comments when possible
- Use descriptive names instead
- Short comments only where absolutely necessary
- Docstrings for public functions and classes

### Code Structure
- Code should read top to bottom like a story
- Functions at the same abstraction level should be together
- Caller functions above callee functions

### Python-Specific
- Use built-in types (no typing library)
- Use list comprehensions over loops where readable
- Prefer stateless functions
- Keep indentation minimal

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

For bug fixes:
```bash
git checkout -b fix/bug-description
```

### 2. Make Your Changes

- Follow the code style guidelines above
- Keep changes focused and atomic
- Write descriptive commit messages

### 3. Test Your Changes

"""bash
python -m pytest tests/ -v
"""

### 4. Update Documentation

- Update README.md if adding new features
- Add docstrings to new functions
- Update relevant documentation in `docs/`

## Pull Request Process

### 1. Before Submitting

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commits are clean and descriptive

### 2. Submit Pull Request

- Provide a clear title and description
- Reference any related issues
- Explain what changes were made and why
- Include screenshots for visual changes

### 3. Review Process

- Maintainers will review your PR
- Address any feedback or requested changes
- Once approved, your PR will be merged

## Types of Contributions

### Bug Fixes
- Report bugs via GitHub Issues
- Include reproduction steps
- Provide system information (OS, Python version, PyTorch version)

### New Features
- Open an issue first to discuss the feature
- Ensure it aligns with the educational goals
- Keep implementations clean and well-documented

### Documentation
- Fix typos or unclear explanations
- Add examples or clarifications
- Improve existing documentation

### Tests
- Add missing test coverage
- Improve existing tests
- Add edge case testing

## Code Review Criteria

Reviewers will check for:

1. **Correctness** - Does the code work as intended?
2. **Clarity** - Is the code easy to understand?
3. **Simplicity** - Is it the simplest solution?
4. **Maintainability** - Can others easily modify it?
5. **Tests** - Are changes properly tested?
6. **Documentation** - Is it well-documented?

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion on GitHub Discussions
- Reach out to maintainers

## Recognition

All contributors will be acknowledged in the project. Thank you for helping make this tutorial better!

