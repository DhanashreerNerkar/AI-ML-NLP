# 1D-Convolution Stencil

A stencil operation is fundamental to many algorithms like discretization methods, interpolation etc.


![image](https://user-images.githubusercontent.com/113553039/221666798-b766976a-3a7a-437d-8612-0bff280951e9.png)

Each output element is the sum of the input elements within a radius. In the example above, the given radius is 3, hence each of the output element is the sum of the 7 input elements. 

In this operation, the stencil function computes the convolution of image and mask to produce the output. The image, mask, output are initialized in managed memory in the host and shared memory is allocated inside the kernel dynamically. The 1D-convolution function is given as:

![image](https://user-images.githubusercontent.com/113553039/221671170-9815e3f3-9f44-488f-a191-32874ae6baac.png)

This function is executed in the file 'stencil.cu' considering appropriate boundary conditions as well.   

                          

