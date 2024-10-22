# DAGGER Wheels

These wheels are the equivalent of `setup.exe` for classic software but for python. You can install them with `pip install nameofwheel.whl`. As `DAGGER` contains `c++` compiled code, they are specific to `python` version and `OS`.

NOTE: These linux wheel are not PEP compliant and _may_ not be compatible with older or basic linux (let's be honest, this is rare or if it happens you'll know how to recompile the tools by yourself). To fix them, you can use `auditwheel`: `auditwheel repair --plat manylinux_2_34_x86_64  daggerpy*linux*`