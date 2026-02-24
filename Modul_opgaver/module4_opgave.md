Exercises/Tasks 4:

Compress your model by post-training quantization e.g. using TensorRT or PyTorch
Benchmark your model in terms of inference time and accuracy after compression.
Implement an inference script that utilizes batch inference with the compressed model
Prune your model, e.g. by gradually removing lowest-magnitude weights, and observe when a significant drop in accuracy occurs. 
Try to recover the "lost" accuracy by fine-tuning a strongly pruned model. 
Include at least one of the inference optimization techniques in your MLOps pipeline
Documentation

In addition to briefly discussing the relevant topics covered in this lecture and detailing how you've applied specific methods in your MLOps project (i.e., by solving the exercises above), your report must also include documentation of the following items.

D4.1: Document the speedup of model compression and any difference in accuracy, and comment on the findings.
D4.2: Document the speedup achieved by batch processing, including considerations on balancing latency and throughput, and noting when the throughput saturates. Document if the throughput is bound by compute or memory bandwidth. 
D4.3: Document how you pruned your model, and show a plot of the degree of pruning vs. accuracy. 
D4.4: Document the effect on the accuracy after fine-tuning the pruned model. 