# rotreps-python
A euclidean transformation representation library

## Basic usage:
```python
from rotreps import transformations as t

# Convert a unit quaternion to yaw, pitch, roll
print(t.quat2euler([0,0,0,1])) # print [ 3.14159, 0, 0 ]
```
