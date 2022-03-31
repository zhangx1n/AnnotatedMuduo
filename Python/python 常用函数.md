## Lambda

```python
>>> add_one = lambda x: x + 1
>>> add_one(2)
3
```

## Map

```python
>>> str_nums = ["4", "8", "6", "5", "3", "2", "8", "9", "2", "5"]

>>> int_nums = map(int, str_nums)
>>> int_nums
<map object at 0x7fb2c7e34c70>

>>> list(int_nums)
[4, 8, 6, 5, 3, 2, 8, 9, 2, 5]

>>> str_nums
["4", "8", "6", "5", "3", "2", "8", "9", "2", "5"]
```

```python
>>> numbers = [1, 2, 3, 4, 5]

>>> squared = map(lambda num: num ** 2, numbers)

>>> list(squared)
[1, 4, 9, 16, 25]
```

## Filter

```python
>>> numbers = [-2, -1, 0, 1, 2]

>>> # Using a lambda function
>>> positive_numbers = filter(lambda n: n > 0, numbers)
>>> positive_numbers
<filter object at 0x7f3632683610>
>>> list(positive_numbers)
[1, 2]

>>> # Using a user-defined function
>>> def is_positive(n):
...     return n > 0
...
>>> list(filter(is_positive, numbers))
[1, 2]
```

## Reduce

```python
>>> def my_add(a, b):
...     result = a + b
...     print(f"{a} + {b} = {result}")
...     return result

>>> my_add(5, 5)
5 + 5 = 10
10

>>> from functools import reduce

>>> numbers = [0, 1, 2, 3, 4]

>>> reduce(my_add, numbers)
0 + 1 = 1
1 + 2 = 3
3 + 3 = 6
6 + 4 = 10
10

>>> from functools import reduce

>>> numbers = [0, 1, 2, 3, 4]

>>> reduce(my_add, numbers, 100)
100 + 0 = 100
100 + 1 = 101
101 + 2 = 103
103 + 3 = 106
106 + 4 = 110
110

>>> from functools import reduce

>>> numbers = [1, 2, 3, 4]

>>> reduce(lambda a, b: a + b, numbers)
10
```

