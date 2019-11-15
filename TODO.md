- create a `CONV -> CONV` onnx graph
   - execute it using onnxruntime, and verify that it produces the correct
     result
   - execute it on the simulator, and verify that it produces the correct result

- create a

```
CONV ---> CONV ---> CONV  ---> ADD
      |                    ^
      |                    |
      +--------------------+
```

onnx graph
   - execute it using onnxruntime
   - execute it on the simulator
