## 一、简介

`argparse`是python用于解析命令行参数和选项的标准模块，用于代替已经过时的optparse模块。`argparse`模块的作用是用于解析命令行参数。

## 二、使用步骤



```python
1：import argparse

2：parser = argparse.ArgumentParser()

3：parser.add_argument()

4：parser.parse_args()
```

解释：
 首先导入该模块；
 然后创建一个解析对象；
 然后向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项；
 最后调用parse_args()方法进行解析；
 解析成功之后即可使用。

## 三、创建解析器对象ArgumentParser



```python
ArgumentParser(prog=None, usage=None,description=None, epilog=None, parents=[],formatter_class=argparse.HelpFormatter, prefix_chars='-',fromfile_prefix_chars=None, argument_default=None,conflict_handler='error', add_help=True)
```

**prog**：程序的名字，默认为sys.argv[0]，用来在help信息中描述程序的名称。
**usage**：描述程序用途的字符串
**description**：help信息前的文字。
**epilog**：help信息之后的信息



```python
import argparse
parse = argparse.ArgumentParser(prog = 'argparseDemo',prefix_chars= '+',description='the message info before help info',
epilog="the message info after help info")
parse.print_help()
```

输出



```python
$ python argparseDemo.py 
usage: argparseDemo [+h]

the message info before help info

optional arguments:
  +h, ++help  show this help message and exit

the message info after help info
```

**add_help**：设为False时，help信息里面不再显示-h --help信息。
**prefix_chars**：参数前缀，默认为'-'



```python
import argparse
parse = argparse.ArgumentParser(prog = 'argparseDemo',prefix_chars= '+')
parse.add_argument('+f')
parse.add_argument('++bar')
print parse.parse_args()
```

输出



```python
 $python argparseDemo.py ++bar 123 +f 123
Namespace(bar='123', f='123')
```

**fromfile_prefix_chars**：前缀字符，放在文件名之前
 **argument_default**：参数的全局默认值。
 **conflict_handler**：对冲突的处理方式，默认为返回错误“error”。还有“resolve”，智能解决冲突。当用户给程序添加了两个一样的命令参数时，“error”就直接报错，提醒用户。而“resolve”则会去掉第一次出现的命令参数重复的部分或者全部（可能是短命令冲突或者全都冲突）。



```bash
import argparse
parse = argparse.ArgumentParser(prog = 'argparseDemo')
parse.add_argument('-f', '--foo', help='old foo help')
parse.add_argument('--foo', help='new foo help')
parse.print_help()
```

输出



```go
argparse.ArgumentError: argument -f/--foo: conflicting option string(s): --foo
```

添加conflict_handler策略后



```bash
parse = argparse.ArgumentParser(prog = 'argparseDemo',conflict_handler = 'resolve')
```

输出



```bash
usage: argparseDemo [-h] [-f FOO] [--foo FOO]

optional arguments:
  -h, --help  show this help message and exit
  -f FOO      old foo help
  --foo FOO   new foo help
```

## 四、add_argument()方法，用来指定程序需要接受的命令参数



```css
add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
```

**name or flags**：参数有两种，可选参数和位置参数。parse_args()运行时，会用'-'来认证可选参数，剩下的即为位置参数。定位参数必选，可选参数可选。
 添加可选参数：



```bash
 parser.add_argument('-f', '--foo')
```

添加位置参数：



```bash
 parser.add_argument('bar')
```



```dart
import argparse
parse = argparse.ArgumentParser()
parse.add_argument('-f', '--foo')
parse.add_argument('bar')
parse.parse_args(['baffr'])
```

输出，可见缺少位置参数'bar'时候，程序报错



```ruby
Namespace(bar='baffr', foo=None)
>>> parse.parse_args(['123','-f','56'])
Namespace(bar='123', foo='56')
>>> parse.parse_args(['-f','56'])
usage: [-h] [-f FOO] bar
: error: too few arguments
shiqqdeMacBook-Pro:Python Demo sqq$ 
```

**action**:参数动作
 argparse内置6种动作可以在解析到一个参数时进行触发：
 `store` 保存参数值，可能会先将参数值转换成另一个数据类型。若没有显式指定动作，则默认为该动作。
 `store_const` 保存一个被定义为参数规格一部分的值，而不是一个来自参数解析而来的值。这通常用于实现非布尔值的命令行标记。



```python
>>> import argparse
>>> parse = argparse.ArgumentParser()
>>> parse.add_argument('--foo', action='store_const', const=42)
_StoreConstAction(option_strings=['--foo'], dest='foo', nargs=0, const=42, default=None, type=None, choices=None, help=None, metavar=None)
>>> parse.parse_args('--foo'.split())
Namespace(foo=42)
```

`store_ture/store_false` 保存相应的布尔值。这两个动作被用于实现布尔开关。



```python
>>> import argparse
>>> parse = argparse.ArgumentParser()
>>> parse.add_argument('-b',action = 'store_true')
_StoreTrueAction(option_strings=['-b'], dest='b', nargs=0, const=True, default=False, type=None, choices=None, help=None, metavar=None)
>>> parse.add_argument('-c',action = 'store_false')
_StoreFalseAction(option_strings=['-c'], dest='c', nargs=0, const=False, default=True, type=None, choices=None, help=None, metavar=None)
>>> parse.parse_args('-b -c'.split())
Namespace(b=True, c=False)
```

`append` 将值保存到一个列表中。若参数重复出现，则保存多个值。



```python
>>> import argparse
>>> parse = argparse.ArgumentParser()
>>> parse.add_argument('-b',action = 'append')
_AppendAction(option_strings=['-b'], dest='b', nargs=None, const=None, default=None, type=None, choices=None, help=None, metavar=None)
>>> parse.parse_args('-b  100 -b 200'.split())
Namespace(b=['100', '200'])
>>> 
```

插入：split()函数 ''-b 100 -b 200''.split() 返回为(['-b','100' ,'-b', '200']

`append_const` 将一个定义在参数规格中的值保存到一个列表中。
 `version` 打印关于程序的版本信息，然后退出



```python
>>> import argparse
>>> parse = argparse.ArgumentParser(prog = 'the demo ')
>>> parse.add_argument('--version',action = 'version',version = '%(prog)s2.0')
_VersionAction(option_strings=['--version'], dest='version', nargs=0, const=None, default='==SUPPRESS==', type=None, choices=None, help="show program's version number and exit", metavar=None)
>>> parse.parse_args('--version'.split())
the demo 2.0
```

`count`统计参数出现的次数



```python
>>> import argparse
>>> parse = argparse.ArgumentParser()
>>> parse.add_argument('-b',action = 'count')
_CountAction(option_strings=['-b'], dest='b', nargs=0, const=None, default=None, type=None, choices=None, help=None, metavar=None)
>>> parse.parse_args('-b -b'.split())
Namespace(b=2)
```

**nargs**：参数的数量
 值可以为整数N(N个)，*(任意多个)，+(一个或更多)
 值为？时，首先从命令行获得参数，若没有则从const获得，然后从default获得：
 **dest**：参数值就保存为parse_args()返回的命名空间对象中名为该 dest 参数值的一个属性。
 如果提供dest，例如dest="a"，那么可以通过args.a访问该参数
 **default**：设置参数的默认值
 **type**：把从命令行输入的结果转成设置的类型
 **choice**：允许的参数值



```bash
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], help="increase output verbosity")
```

**required**：是否必选
 **desk**：可作为参数名
 **help**：参数命令的介绍

## 五、参数有几种写法：

最常见的空格分开：



```python
>>> parser.parse_args('-x X'.split())
Namespace(foo=None, x='X')
>>> parser.parse_args('--foo FOO'.split())
Namespace(foo='FOO', x=None)
```

长选项用=分开 ( “长”选项名字，即选项的名字多于一个字符)



```python
>>> parser.parse_args('--foo=FOO'.split())
Namespace(foo='FOO', x=None)
```

短选项可以写在一起：



```python
>>> parser.parse_args('-xX'.split())
Namespace(foo=None, x='X')
```

感谢
 [Python命令行解析库argparse](https://link.jianshu.com?t=http://www.cnblogs.com/linxiyue/p/3908623.html)



