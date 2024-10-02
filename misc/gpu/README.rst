GPU Utility Scripts
===================

These scripts are intended for debugging and managing JAX use of GPUs:

- ``availgpu.py``: Automatically recommend a setting of the ``CUDA_VISIBLE_DEVICES`` environment variable that excludes GPUs that are already in use.
- ``envinfo.py``: An aid to debugging JAX GPU access.
