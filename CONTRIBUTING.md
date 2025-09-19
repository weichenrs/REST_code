# Contributing to REST

Thank you for your interest in contributing to REST! We welcome contributions from the community to make this project better.

## 🤝 How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear, descriptive title
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - System information (OS, Python version, GPU, etc.)
   - Code snippets or error messages

### Contributing Code

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/REST_code.git
   cd REST_code
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow existing code style
   - Add tests if applicable
   - Update documentation as needed

4. **Test your changes**
   ```bash
   # Run existing tests
   python -m pytest tests/
   
   # Test with sample data
   python tools/test.py configs/rest/rest_water_swin_large.py checkpoints/REST_water_swin_large.pth
   ```

5. **Submit a Pull Request**
   - Provide clear description of changes
   - Reference related issues
   - Ensure CI tests pass

## 📋 Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for new functions and classes
- Keep line length under 120 characters

### Commit Messages

Use clear, descriptive commit messages:
```
feat: add support for custom datasets
fix: resolve memory leak in SPIM module
docs: update installation guide
```

### Testing

- Add tests for new features
- Ensure existing tests pass
- Test on different datasets when possible

## 🎯 Areas for Contribution

We especially welcome contributions in:

- **New dataset support**: Adding more remote sensing datasets
- **Performance optimization**: Improving training/inference speed
- **Documentation**: Tutorials, examples, API documentation
- **Bug fixes**: Addressing reported issues
- **Code quality**: Refactoring, optimization, style improvements

## 📞 Getting Help

- **Questions**: Open a discussion on GitHub
- **Issues**: Check existing issues or create a new one
- **Documentation**: Refer to our comprehensive guides

## 📜 Code of Conduct

Please be respectful and professional in all interactions. We're committed to providing a welcoming environment for everyone.

---

Thank you for helping make REST better! 🚀