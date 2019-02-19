1. Model (Total params: 577,459) load (GeForce GTX 1080 - Intel i7-8700):   
| Name                   | GPU Compute Power | GPU Memory Usage (MB) | CPU Compute Power | RAM Usage (MB) | Time (ms)     |   
| :---------------:      | :---------------: | :-------------------: | :---------------: | :------------: | ------------: |   
| CPU to CPU             | 0                 | 0                     | 0                 | 7              | 11            |   
| CPU to GPU             | 0                 | 443                   | 0                 | 1649           | 1255          |
| GPU to CPU             | 0                 | 0                     | 0                 | 7              | 11            |  
| GPU to GPU             | 0                 | 443                   | 0                 | 1649           | 1255          |

2. Image (1080, 1920, 3) load to Tensor (GeForce GTX 1080 - Intel i7-8700):   
| Name                   | GPU Compute Power | GPU Memory Usage (MB) | CPU Compute Power | RAM Usage (MB) | Time (ms)     |   
| :---------------:      | :---------------: | :-------------------: | :---------------: | :------------: | ------------: |   
| CPU Tensor             | 0                 | 0                     | 0                 | 3              | 39            |   
| CUDA Tensor            | 0                 | 24                    | 0                 | 171            | 41            |

3. Posenet forward (1080, 1920, 3) (GeForce GTX 1080 - Intel i7-8700):   
# Posenet50 - scale_factor = 0.5 - output_stride = 16 - loop measure 10000

| Name                   | GPU Compute Power | GPU Memory Usage (MB) | CPU Compute Power | RAM Usage (GB) | Time (ms)     |   
| :---------------:      | :---------------: | :-------------------: | :---------------: | :------------: | ------------: |   
| Model(CPU) Img(CPU)    | 0                 | 0                     | Full 6 core       | 2.11           | 54.0468       |   
| Model(GPU) Img(GPU)    | 24%               | 543                   | Full 1 core       | 3.63           | 4.3199        |
