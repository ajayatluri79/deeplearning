
#### `Integers and floats work as you would expect from other  languages:`

```python
eip = 3
print(type(eip)) # Prints "<class 'int'>"
print(eip)       # Prints "3"
print(eip + 1)   # Addition; prints "4"
print(eip - 1)   # Subtraction; prints "2"
print(eip * 2)   # Multiplication; prints "6"
print(eip ** 2)  # Eeipponentiation; prints "9"
eip += 1
print(eip)  # Prints "4"
eip *= 2
print(eip)  # Prints "8"
y = 2.5
print(type(y)) # Prints "<class 'float'>"
print(y, y + 1, y * 2, y ** 2) # Prints "2.5 3.5 5.0 6.25"
```

<br>

#### `English words instead of symbols && ||`

```python
mlbr = True
eip = False
print(type(mlbr)) # Prints "<class 'bool'>"
print(mlbr and eip) # Logical AND; prints "False"
print(mlbr or eip)  # Logical OR; prints "True"
print(not mlbr)   # Logical NOT; prints "False"
print(mlbr != eip)  # Logical XOR; prints "True"
```

<br>

#### `Strings in Python`

```python
mlbr = 'hello'    # String literals can use single quotes
eip = "world"    # or double quotes; it does not matter.
print(mlbr)       # Prints "hello"
print(len(mlbr))  # String length; prints "5"
hw = mlbr + ' ' + eip  # String concatenation
print(hw)  # prints "hello world"
hw12 = '%s %s %d' % (mlbr, eip, 12)  # sprintf style string formatting
print(hw12)  # prints "hello world 12"
```

<br>


#### `Useful methods`


```python
eip = "hello"
print(eip.capitalize())  # Capitalize a string; prints "Hello"
print(eip.upper())       # Convert a string to uppercase; prints "HELLO"
print(eip.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
print(eip.center(7))     # Center a string, padding with spaces; prints " hello "
print(eip.replace('l', '(ell)'))  # Replace all instances of one substring with another; # prints "he(ell)(ell)o"
print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"
```

<br>

#### `Python Containers - List`


```python
eip_list = [3, 1, 2]    # Create a list
print(eip_list, eip_list[2])  # Prints "[3, 1, 2] 2"
print(eip_list[-1])     # Negative indices count from the end of the list; prints "2"
eip_list[2] = 'foo'     # Lists can contain elements of different types
print(eip_list)         # Prints "[3, 1, 'foo']"
eip_list.append('bar')  # Add a new element to the end of the list
print(eip_list)         # Prints "[3, 1, 'foo', 'bar']"
x = eip_list.pop()      # Remove and return the last element of the list
print(x, eip_list)      # Prints "bar [3, 1, 'foo']"
```

<br>


#### `Slicing`


```python
eip_list = list(range(5))     # range is a built-in function that creates a list of integers
print(eip_list)               # Prints "[0, 1, 2, 3, 4]"
print(eip_list[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(eip_list[2:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(eip_list[:2])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(eip_list[:])            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
print(eip_list[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
eip_list[2:4] = [8, 9]        # Assign a new sublist to a slice
print(eip_list)               # Prints "[0, 1, 8, 9, 4]"
```

<br>


#### `Loops`

```python
mlbr = ['cat', 'dog', 'monkey']
for eip in mlbr:
    print(eip)
# Prints "cat", "dog", "monkey", each on its own line.
```

<br>

#### `Enumerate`


```python
mlbr = ['cat', 'dog', 'monkey']
for idx, eip in enumerate(mlbr):
    print('#%d: %s' % (idx + 1, eip))
# Prints "#1: cat", "#2: dog", "#3: monkey", each on its own line
```

<br>


#### `List Comprehension`


```python
eip_list = [0, 1, 2, 3, 4]
mlbr_in = []
for mlbr in eip_list:
    mlbr_in.append(mlbr ** 2)
print(mlbr_in)   # Prints [0, 1, 4, 9, 16]

eip_list = [0, 1, 2, 3, 4]
mlbr_in = [mlbr ** 2 for mlbr in eip_list]
print(mlbr_in)   # Prints [0, 1, 4, 9, 16]

# list comprehensions can also contain conditions

eip_list = [0, 1, 2, 3, 4]
mlbr_out = [mlbr ** 2 for mlbr in eip_list if mlbr % 2 == 0]
print(mlbr_out)  # Prints "[0, 4, 16]"
```

<br>



#### `Dictionaries`


```python
eip_dict = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(eip_dict['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in eip_dict)     # Check if a dictionary has a given key; prints "True"
eip_dict['fish'] = 'wet'     # Set an entry in a dictionary
print(eip_dict['fish'])      # Prints "wet"
# print(eip_dict['monkey'])  # KeyError: 'monkey' not a key of eip_dict
print(eip_dict.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(eip_dict.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
del eip_dict['fish']         # Remove an element from a dictionary
print(eip_dict.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"


# It is easy to iterate over the keys in a dictionary

eip_dict = {'person': 2, 'cat': 4, 'spider': 8}
for animal in eip_dict:
    legs = eip_dict[animal]
    print('A %s has %eip_dict legs' % (animal, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
```

<br>


#### `Dictionary Comprehension`


```python
# If you want access to keys and their corresponding values, use the items method:

eip_dict = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in eip_dict.items():
    print('A %s has %d legs' % (animal, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"

# Dictionary comprehension

cip_list = [0, 1, 2, 3, 4]
mlbr_out = {x: x ** 2 for x in cip_list if x % 2 == 0}
print(mlbr_out)  # Prints "{0: 0, 2: 4, 4: 16}"
```

<br>


####  `Sets`


```python
mlbr_sets = {'cat', 'dog'}
print('cat' in mlbr_sets)   # Check if an element is in a set; prints "True"
print('fish' in mlbr_sets)  # prints "False"
mlbr_sets.add('fish')       # Add an element to a set
print('fish' in mlbr_sets)  # Prints "True"
print(len(mlbr_sets))       # Number of elements in a set; prints "3"
mlbr_sets.add('cat')        # Adding an element that is already in the set does nothing
print(len(mlbr_sets))       # Prints "3"
mlbr_sets.remove('cat')     # Remove an element from a set
print(len(mlbr_sets))       # Prints "2"
```

<br>


#### `Tuples`


```python
eip_dict = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
print (eip_dict)
mlbr_tuples = (5, 6)        		# Create a tuple
print(type(mlbr_tuples))			# Prints "<class 'tuple'>"
print(eip_dict[mlbr_tuples])		# Prints "5"
print(eip_dict[(1, 2)])  			# Prints "1"
```

<br>


#### `Functions`


```python
def mlbr(eip):
    if eip > 0:
        return 'positive'
    elif eip < 0:
        return 'negative'
    else:
        return 'zero'

for eip in [-1, 0, 1]:
    print(mlbr(eip))
# Prints "negative", "zero", "positive"
```
<br>

#### `Function Arguments`

```python
def eip_func(name, loud=False):
    if loud:
        print('HELLO, %s!' % name.upper())
    else:
        print('Hello, %s' % name)

eip_func('Bob') # Prints "Hello, Bob"
eip_func('Fred', loud=True)  # Prints "HELLO, FRED!"
```

<br>

#### `NumPy`

```python
import numpy as np

mlbr_in = np.array([1, 2, 3])   # Create a rank 1 array
print(type(mlbr_in))            # Prints "<class 'numpy.ndarray'>"
print(mlbr_in.shape)            # Prints "(3,)"
print(mlbr_in[0], mlbr_in[1], mlbr_in[2])   # Prints "1 2 3"
mlbr_in[0] = 5                  # Change an element of the array
print(mlbr_in)                  # Prints "[5, 2, 3]"

mlbr_out = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(mlbr_out.shape)                     # Prints "(2, 3)"
print(mlbr_out[0, 0], mlbr_out[0, 1], mlbr_out[1, 0])   # Prints "1 2 4"
```

<br>

#### `Some NumPy Functions`

```python
import numpy as np

mlblr = np.zeros((2,2))   # Create an array of all zeros
print(mlblr)                    # Prints "[[ 0.  0.]
						                         #             [ 0.  0.]]"

eip = np.ones((1,2))      # Create an array of all ones
print(eip)                     # Prints "[[ 1.  1.]]"

mlblr_in = np.full((2,2), 7)  # Create a constant array
print(mlblr_in)               # Prints "[[ 7.  7.]
							                      #             [ 7.  7.]]"

mlblr_out = np.random.random((2,2))  # Create an array filled with random values
print(mlblr_out)                     # Might print "[[ 0.91940167  0.08143941]
									                          #                      [ 0.68744134  0.87236687]]"

```

<br>

#### `Mixing integer indexing with slice indexing`

```python
# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
eip_arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
row_r1 = eip_arr[1, :]    # Rank 1 view of the second row of a
row_r2 = eip_arr[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"
```

<br>


#### `Datatypes`

```python
import numpy as np

eip_arr = np.array([1, 2])   # Let numpy choose the datatype
print(eip_arr.dtype)         # Prints "int64"

eip_arr = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(eip_arr.dtype)             # Prints "float64"

eip_arr = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(eip_arr.dtype)                         # Prints "int64"
```

<br>

#### `Array Math`

```python
import numpy as np

eip_arr = np.array([[1,2],[3,4]], dtype=np.float64)
mlblr_arr = np.array([[5,6],[7,8]], dtype=np.float64)

print(np.add(eip_arr, mlblr_arr)) # or print(eip_arr + mlblr_arr)
# [[ 6.0  8.0]
#  [10.0 12.0]]

print(np.subtract(eip_arr, mlblr_arr)) or print(eip_arr - mlblr_arr)
# [[-4.0 -4.0]
#  [-4.0 -4.0]]

print(np.multiply(eip_arr, mlblr_arr)) or print(eip_arr * mlblr_arr)
# [[ 5.0 12.0]
#  [21.0 32.0]]

print(np.divide(eip_arr, mlblr_arr)) or print(eip_arr / mlblr_arr)
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]

print(np.sqrt(eip_arr))
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
```

<br>

#### `Dot function`

```python
import numpy as np

eip_arr = np.array([[1,2],[3,4]])
mlblr_arr = np.array([[5,6],[7,8]])

mlblr_in = np.array([9,10])
mlblr_out = np.array([11, 12])

# Inner product of vectors; both produce 219
print(mlblr_in.dot(mlblr_out))
print(np.dot(mlblr_in, mlblr_out))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(eip_arr.dot(mlblr_in))
print(np.dot(eip_arr, mlblr_in))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(eip_arr.dot(mlblr_arr))
print(np.dot(eip_arr, mlblr_arr))

```

<br>

#### `Sum along an axis`

```python
import numpy as np

eip_arr = np.array([[1,2],[3,4]])

print(np.sum(eip_arr))  # Compute sum of all elements; prints "10"
print(np.sum(eip_arr, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(eip_arr, axis=1))  # Compute sum of each row; prints "[3 7]"
```

<br>

#### `MathPlotLib - Plotting Graphs`

```python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
eip = np.arange(0, 3 * np.pi, 0.1)
mlblr = np.sin(eip)

# Plot the points using matplotlib
plt.plot(eip, mlblr)
plt.show()  # You must call plt.show() to make graphics appear.
```

<br>


#### `Images`


```python
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

# Uncomment the line below if you're on a notebook
# %matplotlib inline 
mlblr_img = imread('assets/cat.jpg')
img_tinted = mlblr_img * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(mlblr_img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()
```

<br>
