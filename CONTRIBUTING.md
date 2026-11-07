# Contributing

Thank you for your interest in contributing to **Signal Integrity Analyzer**! Contributions of all sizes are welcome, including bug fixes, analysis improvements, tests, documentation, and usability enhancements.

## Before you start

- Read the project [README](README.md) to understand its purpose and current capabilities.
- Check the existing issues and pull requests to avoid duplicating ongoing work.
- For a significant feature or architectural change, open an issue first to discuss the proposed approach.
- Keep changes focused. A pull request should ideally address one problem or introduce one coherent feature.

## Reporting bugs

When opening a bug report, include as much of the following information as possible:

- A clear description of the problem and the behavior you expected.
- Steps to reproduce the issue.
- Your operating system and Python version.
- The relevant application settings and analysis parameters.
- A small, non-confidential `.s2p` sample that reproduces the issue, if you can share one.
- The full error message or traceback, where applicable.
- Screenshots of incorrect plots, if they help explain the issue.

Do not upload confidential, proprietary, or personally sensitive measurement data.

## Suggesting features

Feature requests should explain the use case, the expected behavior, and why the feature would be useful for signal-integrity analysis. For changes to numerical methods, include references, equations, or test cases where possible.

## Development workflow

1. Fork the repository.
2. Clone your fork and create a dedicated branch:
   ```bash
   git clone https://github.com/<your-username>/Signal-Integrity-Analyzer.git
   cd Signal-Integrity-Analyzer
   git switch -c feature/short-description
   ```
3. Set up a Python environment and install the project's dependencies using the dependency instructions provided in the repository.
4. Make your changes. Keep the code readable and consistent with the existing project style.
5. Run the relevant checks and test your changes with representative `.s2p` data.
6. Commit your work with a clear message:
   ```bash
   git add .
   git commit -m "Add: short description of change"
   ```
7. Push your branch and open a Pull Request against the repository's `main` branch.

## Guidelines for code contributions

- Preserve existing behavior unless the change is intentional and documented.
- Prefer clear names, small functions, and useful comments over overly complex implementations.
- Validate input data and handle malformed or incomplete Touchstone files gracefully.
- Keep frequency units, reference impedance, port ordering, and S-parameter conventions explicit.
- For numerical or signal-processing changes, explain the method and its assumptions.
- Avoid presenting reconstructed eye/jitter results as measured results unless they actually come from measurements.
- Do not commit generated files, local environments, caches, or private measurement data unless they are intentionally required by the project.

## Testing and validation

For changes that affect parsing, calculations, or plots, please:

- Test with at least one valid two-port `.s2p` file.
- Where possible, test edge cases such as malformed headers, unexpected frequency units, missing data, and different frequency ranges.
- Compare numerical results against a trusted reference or independently calculated example when practical.
- Include a regression test for bugs whenever the project has an appropriate test structure.
- Include before/after screenshots for meaningful visual changes.

If a test cannot be run, state that clearly in the pull request rather than implying that it passed.

## Pull Request checklist

Before submitting, check that:

- [ ] The pull request has a clear title and explains the motivation for the change.
- [ ] The changes are limited to the intended scope.
- [ ] Relevant tests or manual validation have been performed.
- [ ] New dependencies, if any, are justified and documented.
- [ ] Documentation and screenshots have been updated where needed.
- [ ] No private data, credentials, or unrelated generated files are included.

## Questions

If you are unsure whether an idea fits the project, open an issue for discussion before investing significant development time.

Thank you for helping improve Signal Integrity Analyzer!
