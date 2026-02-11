Exercises/Tasks:

Implement a training script (train_ddp.py) that scales the training using data parallelism e.g. using PyTorch Distributed Data Parallel (DDP) across multiple GPUs on the same node (Use AI-Lab)
Implement a memory optimization strategy e.g. Automatic Mixed Precision (AMP).
Scale the training with data parallelism across multiple nodes e.g. using torchrun or DeepSpeed.
Implement the ZeRO optimizer using DeepSpeed and experiment with the different stages to save VRAM.
Include at least one of the optimization techniques above in your MLOps pipeline.
Remember to use separate branches when experimenting with new features in your MLOps project pipeline and merge when functionality and unit tests are in place.

Note that you do not have to to include distributed training in your MLOps pipelines, although it is possible to automate running a job on AI-Lab.

Documentation

In addition to briefly discussing the relevant topics covered in this lecture and detailing how you've applied specific methods in your MLOps project (i.e., by solving the exercises above), your report must also include documentation of the following items.

D3.1: An estimate of the speedup of your model training by parallelization
D3.2: An estimate of the scaling needed (compute, dataset, parameters) to halve your models current test loss following the power-law (It's okay to use the constants found for LLMs even though your model might not be a LLM)
D3.3: The effect of the implemented multi-GPU parallelization strategy e.g. training time speed-up
D3.4: The effect of the implemented multi-node parallelization strategy e.g. training time speed-up
D3.5: The effect of the implemented memory optimization strategies e.g. VRAM savings and effect on training time using AMP. 
D3.6: The effect of the different stages of the ZerO optimizer in relation to VRAM savings and effect on training time. 
