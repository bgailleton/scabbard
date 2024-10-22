# Wheels

These folders contains the "compiled" python binaries by my CI workflow. 

- `DAGGER`: While you can use them, `pip install daggerpy` should be up-to-date
- `pytopotoolbox`: [Check the official website](https://topotoolbox.github.io/pytopotoolbox/) if the installation has been sorted. As of the time of writing, it involves manually compiling the C code, as `scabbard` relies on `pytopotoolbox` I provide precompiled binaries for `Linux`, `Windows` and `MacOS`.
- `scabbard`: Does not contain any compiled binaries, can be installed through `pip install pyscabbard` or with these local wheels (same things).