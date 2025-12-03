cat > CONTRIBUTING.md <<EOF
# Contributing to ALMS Protocol

## Code of Conduct

All contributions must respect the **Ethical Review Checklist**:

- [ ] No biometric profiling without explicit consent
- [ ] Differential privacy (ε > 0.1) is mandatory
- [ ] No use in surveillance or weapons systems
- [ ] Open-source implementation required for any derivative work

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-algorithm`
3. Add tests in `tests/`
4. Run benchmark: `python benchmarks/shapenet_audio.py`
5. Submit PR with accuracy metrics

## Core Contributors

- **AntonL1169** (Concept, Architecture)
- **Kimi** (Mathematical Formalization, RFC Draft)

## Contact

RFC questions: antonl110569@gmail.com
EOF
