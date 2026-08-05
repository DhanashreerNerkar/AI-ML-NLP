# Inclusive Scan using CUDA

The task2.cu file creates and fills an array using managed memory and calls a function called "scan1.cu". The scan function proves useful in many applications like building histograms, data structures like trees etc. This scan kernel function implements an inclusive scan using the parallel
Hillis-Steele algorithm. An example of an inclusive scan looks like: 
                        
                                    [3 1 7 0 4 1 6 3]
                                  [3 4 11 11 15 16 22 25]
The Hillis-Steele algorithm takes X steps based on the assumption that the number of elements in the array, say n, follows the formula: n = 2<sup>X </sup>.  In the previous example, we have n = 8, hence X = 3 (2<sup>3 </sup> = 8).                               

                      

![image](https://user-images.githubusercontent.com/113553039/220793962-522d3d7d-668e-4fd6-97ad-f96f08582ebe.png)


We first take the last element of the array, apply the stride (2<sup>0 </sup> in the first iteration) to all the array elements before the last element to find another element to add with each of them. For the second iteration, the stride is 2 = 2<sup>1 </sup> and for the third iteration it is, 4 = 2<sup>2 </sup>. In general, the m<sup>th </sup> iteration has a stride of 2<sup>m - 1 </sup>. 

In order to find the number of addition operations:
Taking the total number of elements as p, in the first iteration we see there are: 2<sup>p </sup> - 2<sup>0 </sup> additions
                                                                 Second iteration: 2<sup>p </sup> - 2<sup>1 </sup> additions
                                                                  Third iteration: 2<sup>p </sup> - 2<sup>2 </sup> additions
                                                                  
In general, the algorithm would perform (2<sup>p </sup> - 2<sup>0 </sup>) + (2<sup>p </sup> - 2<sup>1 </sup>) + .....+ (2<sup>p </sup> - 2<sup> p - 1 </sup>) 

p * 2<sup>p </sup> - (2<sup>0 </sup> + ... + 2<sup> p - 1 </sup>) = p * 2<sup>p </sup> - 2<sup>p </sup> + 1 = n * (log<sub>2</sub>(n) - 1) + 1 

Hence the algorithm requires O(n * log<sub>2</sub>(n)) additions.
