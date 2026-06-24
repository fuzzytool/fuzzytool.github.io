# Citing & releasing

## Citing

If you use fuzzytool in academic work, please cite it. The archival DOI (all
versions) is [10.5281/zenodo.20836712](https://doi.org/10.5281/zenodo.20836712);
each release also mints its own versioned DOI. Repository citation metadata lives
in [`CITATION.cff`](https://github.com/fuzzytool/fuzzytool.github.io/blob/main/CITATION.cff).

## Releasing (maintainers)

Releases are automated. Pushing a version tag triggers
[`release-pypi.yml`](https://github.com/fuzzytool/fuzzytool.github.io/blob/main/.github/workflows/release-pypi.yml),
which builds the sdist + wheel and publishes to PyPI via **Trusted Publishing**
(OIDC — no API token stored).

```bash
# 1. Bump the version in pyproject.toml, fuzzytool/__init__.py, CITATION.cff.
# 2. Update CHANGELOG.md.
git tag v0.1.0
git push --tags
```

One-time setup:

- **PyPI:** register a trusted publisher for the `fuzzytool` project pointing to
  the repo `fuzzytool/fuzzytool.github.io` and workflow `release-pypi.yml`
  (PyPI → project → Publishing).
- **Zenodo:** enable the repository at <https://zenodo.org/account/settings/github/>;
  the next GitHub release is archived automatically and gets a DOI. Repository
  metadata is provided in [`.zenodo.json`](https://github.com/fuzzytool/fuzzytool.github.io/blob/main/.zenodo.json).
