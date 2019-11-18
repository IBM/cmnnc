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

# Steps
 ✓ move weights into initializer for generated onnx models

# Issues
 - batching: CONV operator's first dimension is batch size
 - multiple input nodes (residuals)

 # Multiple input nodes (residuals)

 Let's consider:

 ```
 CONV1D ---> CONV1D ---> ADD
         |           ^
         |           |
         +---------- +
```

Partition P1:
 - CONV1D
    - INPUT:  X
    - OUTPUT: Y, Z1

Partition P2:
 - CONV1D
    - INPUT:  Yl
    - OUTPUT: Z2
 - ADD:
    - INPUT: Z1, Z2
    - OUTPUT: W


