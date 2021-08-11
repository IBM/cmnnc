# Computational Memory Neural Network Compiler (cmnnc)

See [our paper](https://arxiv.org/abs/2003.04293) and our [blog post](https://kkourt.io/blog/2020/04-26-cmnnc.html)
for more info.

## Installation

We use `virtualenv` to install python dependencies:

```
$ virtualenv -p python3 cmenv
$ source cmenv/bin/activate
$ pip install islpy astor onnxruntime numpy onnx pytest z3-solver graphviz
```

Note: you might have to install `libpython3.x-dev` or the equivalent package to
for islpy installation to work.
