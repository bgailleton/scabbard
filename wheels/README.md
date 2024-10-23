# Wheels

These directories contain the precompiled Python binaries from my CI workflow.

- **DAGGER**: Although the binaries are available here, it’s recommended to use `pip install daggerpy` for the latest version.
  
- **pytopotoolbox**: [Check the official website](https://topotoolbox.github.io/pytopotoolbox/) for installation updates. Currently, manual compilation of the C code is required. Since `scabbard` depends on `pytopotoolbox`, I provide precompiled binaries for `Linux`, `Windows`, and `MacOS`.

- **scabbard**: This package doesn’t include compiled binaries. You can install it either through `pip install pyscabbard` or use the provided local wheels (both options are equivalent).
