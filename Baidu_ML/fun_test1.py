import paddle.fluid as fluid
import numpy as np

with fluid.dygraph.guard():
    value = np.arange(26).reshape(2, 13).astype("float32")
    print("value",value)
    a = fluid.dygraph.to_variable(value)
    linear = fluid.Linear(13, 1, dtype="float32")
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01,
                                  parameter_list=linear.parameters())
    out = linear(a)
    print("out",out)
    out.backward()
    optimizer.minimize(out)
    optimizer.clear_gradients()