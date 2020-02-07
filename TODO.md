✓ create a `CONV -> CONV` onnx graph
   ✓ execute it using onnxruntime, and verify that it produces the correct
     result
   ✓ execute it on the simulator, and verify that it produces the correct result

✓ create a

```
CONV ---> CONV ---> ADD
      |              ^
      |              |
      +--------------+
```

onnx graph
   ✓ execute it using onnxruntime
   ✓ execute it on the simulator
   - use a batch size which is not 1

# Steps
 ✓ move weights into initializer for generated onnx models
 - add a check that we do not overwrite something before it is read, and a
   corresponding failing test case

# Issues
 ✓ multiple input nodes (residuals)
 - batching: CONV operator's first dimension is batch size
 - GCU, input/output (DMA)
 - overwrite protection (WAR dependencies)
