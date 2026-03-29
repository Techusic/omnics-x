# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :☑️: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in omicsx, please email raghavmkota@gmail.com with subject line "[SECURITY] omicsx Vulnerability" with:

1. **Description**: Clear explanation of the vulnerability
2. **Affected Versions**: Which versions are impacted
3. **Proof of Concept**: Steps to reproduce or code example
4. **Impact**: Potential impact of the vulnerability
5. **Fix**: If you have a proposed fix

**Please do not:**
- Open public GitHub issues for security vulnerabilities
- Discuss vulnerabilities in pull requests
- Report via GitHub security advisories initially

## Security Considerations

### Memory Safety
- omicsx leverages Rust's memory safety guarantees
- No unsafe code except in SIMD kernel modules with careful review
- All public APIs are memory-safe

### Input Validation
- Protein sequences validated against IUPAC codes
- Matrices validated for correct dimensions and values
- Gap penalties validated to be negative

### Performance Timing
- Algorithms use constant-time operations where possible
- No data-dependent branches in cryptographic operations (if any)
- Side-channel attacks unlikely due to deterministic algorithms

### Dependencies
- Dependencies kept minimal
- Regular updates for known vulnerabilities
- Audit trail maintained via package.lock

## Security Best Practices

1. **Keep Updated**: Always use the latest stable version
2. **Validate Input**: Check sequence data from untrusted sources
3. **Sandboxing**: Use in sandboxed environments if processing untrusted data
4. **Reporting**: Report vulnerabilities responsibly

## Response Timeline

- **Initial Response**: Within 48 hours
- **Assessment**: Within 1 week
- **Fix**: High severity within 2 weeks
- **Disclosure**: After patch release or 90 days, whichever is first

## Security Updates

Security fixes are released in:
1. Patch releases (0.1.1, 0.1.2, etc.)
2. With a security advisory on GitHub
3. Announced via CHANGELOG.md

Thank you for helping keep omicsx secure!
