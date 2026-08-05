# thrust and cub

Thrust is a parallel algorithms library which resembles the Standard Template Library (STL) in C++. The reduce feature in thrust is used in this code to compute the sum of all elements in the range [first, last).

Cub is another library like thrust. The DeviceReduce::Sum feature is used to compute the sum of the numbers given as input. 


The plot below tells the time taken to print the result vs the size of the input array (2<sup>10 </sup>, 2<sup>11 </sup>......2<sup>30 </sup>). It can be seen that cub performs better compared to thrust.

![image](https://user-images.githubusercontent.com/113553039/231063991-28670490-07e1-4420-b447-50644eb1234c.png)




