# Computational Memory Neural Network Compiler (cmnnc)

## Installation

We use `virtualenv` to install python dependencies:

```
$ virtualenv -p python3 cmenv
$ source cmenv/bin/activate
$ pip install islpy astor onnxruntime numpy onnx pytest
```

Note: you might have to install `libpython3.x-dev` or the equivalent package to
for islpy installation to work.
