# Optimizing Parallel Reduction

Assuming we have really huge arrays like 2<sup>10 </sup> or more entries, a tree-based approach is used within each block of threads to sum all entries in such an array. 
  
  
  ![image](https://user-images.githubusercontent.com/113553039/230997372-6447ac6c-4300-484a-9608-ebcea062162d.png)

  
  Multiple thread blocks reduce a chunk of the given array to one single value. To synchronize these thread blocks, multiple kernel launches are used here. In this type of reduction ("First Add During Load"), for the 1st iteration only half of the blocks are needed and replaces single load. Each thread will load an element from global to shared memory and the first level of reduction is performed while loading to increase thread utilization. 
  
The idea of this kind of reduction and the code is inspired from this source:  
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
