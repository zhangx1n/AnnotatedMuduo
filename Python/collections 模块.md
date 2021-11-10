## Counter

Counter是一个dict子类，主要是用来对你访问的对象的频率进行计数。
常用方法：

- elements()：返回一个迭代器，每个元素重复计算的个数，如果一个元素的计数小于1,就会被忽略。
- most_common([n])：返回一个列表，提供n个访问频率最高的元素和计数
- subtract([iterable-or-mapping])：从迭代对象中减去元素，输入输出可以是0或者负数
- update([iterable-or-mapping])：从迭代对象计数元素或者从另一个 映射对象 (或计数器) 添加。

```python
# 统计字符出现的次数
>>> import collections
>>> collections.Counter('hello world')
Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})
# 统计单词数
>>> collections.Counter('hello world hello world hello nihao'.split())
Counter({'hello': 3, 'world': 2, 'nihao': 1})
```

常用的方法：

```python
>>> c = collections.Counter('hello world hello world hello nihao'.split())
>>> c
Counter({'hello': 3, 'world': 2, 'nihao': 1})
# 获取指定对象的访问次数，也可以使用get()方法
>>> c['hello']
3
>>> c = collections.Counter('hello world hello world hello nihao'.split())
# 查看元素
>>> list(c.elements())
['hello', 'hello', 'hello', 'world', 'world', 'nihao']
# 追加对象，或者使用c.update(d)
>>> c = collections.Counter('hello world hello world hello nihao'.split())
>>> d = collections.Counter('hello world'.split())
>>> c
Counter({'hello': 3, 'world': 2, 'nihao': 1})
>>> d
Counter({'hello': 1, 'world': 1})
>>> c + d
Counter({'hello': 4, 'world': 3, 'nihao': 1})
# 减少对象，或者使用c.subtract(d)
>>> c - d
Counter({'hello': 2, 'world': 1, 'nihao': 1})
# 清除
>>> c.clear()
>>> c
Counter()
```

## defaultdict

`collections.defaultdict(default_factory)`为字典的没有的key提供一个默认的值。参数应该是一个函数，当没有参数调用时返回默认值。如果没有传递任何内容，则默认为None。

```python
>>> d = collections.defaultdict()
>>> d
defaultdict(None, {})
>>> e = collections.defaultdict(str)
>>> e
defaultdict(<class 'str'>, {})
```

defaultdict的一个典型用法是使用其中一种内置类型(如str、int、list或dict)作为默认工厂，因为这些内置类型在没有参数调用时返回空类型。

```python
>>> d = collections.defaultdict(str)
>>> d
defaultdict(<class 'str'>, {})
>>> d['hello']
''
>>> d
defaultdict(<class 'str'>, {'hello': ''})
# 普通字典调用不存在的键时，将会抛异常
>>> e = {}
>>> e['hello']
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'hello'
```

使用`int`作为default_factory的例子：

```python
>>> from collections import defaultdict
>>> fruit = defaultdict(int)
>>> fruit['apple'] += 2 
>>> fruit
defaultdict(<class 'int'>, {'apple': 2})
>>> fruit
defaultdict(<class 'int'>, {'apple': 2})
>>> fruit['banana']  # 没有对象时，返回0
0
>>> fruit
defaultdict(<class 'int'>, {'apple': 2, 'banana': 0})
```

使用`list`作为default_factory的例子：

```python
>>> s = [('NC', 'Raleigh'), ('VA', 'Richmond'), ('WA', 'Seattle'), ('NC', 'Asheville')]
>>> d = collections.defaultdict(list)
>>> for k,v in s:
...      d[k].append(v)
... 
>>> d
defaultdict(<class 'list'>, {'NC': ['Raleigh', 'Asheville'], 'VA': ['Richmond'], 'WA': ['Seattle']})
```

## OrderedDict

Python字典中的键的顺序是任意的:它们不受添加的顺序的控制。
`collections.OrderedDict`类提供了保留他们添加顺序的字典对象。

```python
>>> from collections import OrderedDict
>>> o = OrderedDict()
>>> o['key1'] = 'value1'
>>> o['key2'] = 'value2'
>>> o['key3'] = 'value3'
>>> o
OrderedDict([('key1', 'value1'), ('key2', 'value2'), ('key3', 'value3')])
```

如果在已经存在的key上添加新的值，将会保留原来的key的位置，然后覆盖value值。

```python
>>> o['key1'] = 'value5'
>>> o
OrderedDict([('key1', 'value5'), ('key2', 'value2'), ('key3', 'value3')])
```

## namedtuple

三种定义命名元组的方法：第一个参数是命名元组的构造器（如下的：Person，Human）

```python
>>> from collections import namedtuple
>>> Person = namedtuple('Person', ['age', 'height', 'name'])
>>> Human = namedtuple('Human', 'age, height, name')
>>> Human2 = namedtuple('Human2', 'age height name')
```

实例化命令元组

```python
>>> tom = Person(30,178,'Tom')
>>> jack = Human(20,179,'Jack')
>>> tom
Person(age=30, height=178, name='Tom')
>>> jack
Human(age=20, height=179, name='Jack')
>>> tom.age #直接通过  实例名+.+属性 来调用
30
>>> jack.name
'Jack'
```

## deque

`collections.deque`返回一个新的双向队列对象，从左到右初始化(用方法 append()) ，从 iterable （迭代对象) 数据创建。如果 iterable 没有指定，新队列为空。
		`collections.deque`队列支持线程安全，对于从两端添加(append)或者弹出(pop)，复杂度O(1)。
		虽然`list`对象也支持类似操作，但是这里优化了定长操作（pop(0)、insert(0,v)）的开销。
		如果 maxlen 没有指定或者是 None ，deques 可以增长到任意长度。否则，deque就限定到指定最大长度。一旦限定长度的deque满了，当新项加入时，同样数量的项就从另一端弹出。
		支持的方法：

- append(x)：添加x到右端
- appendleft(x)：添加x到左端
- clear()：清楚所有元素，长度变为0
- copy()：创建一份浅拷贝
- count(x)：计算队列中个数等于x的元素
- extend(iterable)：在队列右侧添加iterable中的元素
- extendleft(iterable)：在队列左侧添加iterable中的元素，注：在左侧添加时，iterable参数的顺序将会反过来添加
- index(x[,start[,stop]])：返回第 x 个元素（从 start 开始计算，在 stop 之前）。返回第一个匹配，如果没找到的话，升起 ValueError 。
- insert(i,x)：在位置 i 插入 x 。注：如果插入会导致一个限长deque超出长度 maxlen 的话，就升起一个 IndexError 。
- pop()：移除最右侧的元素
- popleft()：移除最左侧的元素
- remove(value)：移去找到的第一个 value。没有抛出ValueError
- reverse()：将deque逆序排列。返回 None 。
- maxlen：队列的最大长度，没有限定则为None。

```python
>>> from collections import deque
>>> d = deque(maxlen=10)
>>> d
deque([], maxlen=10)
>>> d.extend('python')
>>> [i.upper() for i in d]
['P', 'Y', 'T', 'H', 'O', 'N']
>>> d.append('e')
>>> d.appendleft('f')
>>> d
deque(['f', 'p', 'y', 't', 'h', 'o', 'n', 'e'], maxlen=10)
```

## ChainMap

一个 ChainMap 将多个字典或者其他映射组合在一起，创建一个单独的可更新的视图。 如果没有 maps 被指定，就提供一个默认的空字典 。`ChainMap`是管理嵌套上下文和覆盖的有用工具。

```python
>>> from collections import ChainMap
>>> d1 = {'apple':1,'banana':2}
>>> d2 = {'orange':2,'apple':3,'pike':1}
>>> combined_d = ChainMap(d1,d2)
>>> reverse_combind_d = ChainMap(d2,d1)
>>> combined_d 
ChainMap({'apple': 1, 'banana': 2}, {'orange': 2, 'apple': 3, 'pike': 1})
>>> reverse_combind_d
ChainMap({'orange': 2, 'apple': 3, 'pike': 1}, {'apple': 1, 'banana': 2})
>>> for k,v in combined_d.items():
...      print(k,v)
... 
pike 1
apple 1
banana 2
orange 2
>>> for k,v in reverse_combind_d.items():
...      print(k,v)
... 
pike 1
apple 3
banana 2
orange 2
```

