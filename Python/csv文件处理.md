csv文件（`Comma-Separated Values`）是一种以逗号作为分隔符（当然也可以以其他字符作为分隔符）、以行为数据单位的纯文本数据文件，比如像下面这样的一个文件`data.csv`：



```objectivec
id,name,age,score
1001,Tom,21,89
1005,Jim,23,100
1002,张三,19,78
1003,Jane,20,93
1004,李四,24,94
```

`data.csv`用Excel也可以打开，打开的效果是这样的：

![img](https:////upload-images.jianshu.io/upload_images/8819542-2b7140db57354f8d.png?imageMogr2/auto-orient/strip|imageView2/2/w/324/format/webp)


 发现中文是乱码，那是因为`data.csv`文件的编码格式是utf-8而不是GBK，转为GBK即可在Excel中正常显示中文。

## Reading CSV file with csv.reader()

该[csv.reader()](https://links.jianshu.com/go?to=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fcsv.html%23csv.reader)方法返回一个reader对象，该对象将遍历给定CSV文件中的行。

假设我们有以下numbers.csv包含数字的文件：

6,5,3,9,8,6,7

以下python脚本从此CSV文件读取数据。



```python
#!/usr/bin/python3

import csv
open('numbers.csv', 'r') with f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
```

在上面的代码示例中，我们打开了numbers.csv以读取并使用csv.reader()方法加载数据。

现在，假设CSV文件将使用**其他定界符**。（严格来说，这不是CSV文件，但是这种做法很常见。）例如，我们有以下items.csv文件，其中的元素由竖线字符（|）分隔：



```ruby
pen|table|keyboard
```

以下脚本从items.csv文件读取数据。



```python
#!/usr/bin/python3

import csv
f = open('items.csv', 'r')
with f:
    reader = csv.reader(f, delimiter="|")
    for row in reader:
        for e in row:
            print(e)
```

我们delimiter在csv.reader()方法中使用参数指定新的分隔字符。

## Reading CSV file with csv.DictReader

该[csv.DictReader](https://links.jianshu.com/go?to=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fcsv.html%23csv.DictReader)班的运作就像**一个普通的reader，但读入字典中的信息映射**。

字典的键可以与fieldnames参数一起传递，也可以从CSV文件的第一行推断出来。

我们有以下values.csv文件：



```swift
min, avg, max
1, 5.5, 10
```

第一行代表字典的键，第二行代表值。



```python
#!/usr/bin/python3

import csv
f = open('values.csv', 'r')
with f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)
```

上面的python脚本使用读取values.csv文件中的值csv.DictReader。

这是示例的输出。



```ruby
$ ./read_csv3.py 
{' max': ' 10', 'min': '1', ' avg': ' 5.5'}
```

## Writing CSV file using csv.writer()

该[csv.writer()](https://links.jianshu.com/go?to=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fcsv.html%23csv.writer)方法返回一个writer对象，该对象负责将用户数据转换为给定文件状对象上的定界字符串。



```jsx
#!/usr/bin/python3

import csv
nms = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
f = open('numbers2.csv', 'w')
with f:
    writer = csv.writer(f)
    for row in nms:
        writer.writerow(row)
```

该脚本将数字写入numbers2.csv文件。该writerow()方法将一行数据写入指定的文件。

该脚本将产生以下文件（numbers2.csv）：

1,2,3,4,5,6 7,8,9,10,11,12

可以一次写入所有数据。该writerows()方法将所有给定的行写入CSV文件。

下一个代码示例将Python列表写入numbers3.csv文件。该脚本将三行数字写入文件。



```jsx
#!/usr/bin/python3

import csv
nms = [[1, 2, 3], [7, 8, 9], [10, 11, 12]]
f = open('numbers3.csv', 'w')
with f:
    writer = csv.writer(f)
    writer.writerows(nms)
```

运行上述程序时，以下输出将写入numbers3.csv文件：

1,2,3 7,8,9 10,11,12

## Quoting

可以在CSV文件中引用单词。Python CSV模块中有**四种不同的引用模式**：

- QUOTE_ALL —引用所有字段
- QUOTE_MINIMAL-仅引用那些包含特殊字符的字段
- QUOTE_NONNUMERIC —引用所有非数字字段
- QUOTE_NONE —不引用字段

在下一个示例中，我们向items2.csv文件写入三行。所有非数字字段都用引号引起来。



```jsx
#!/usr/bin/python3

import csv
f = open('items2.csv', 'w')
with f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    writer.writerows((["coins", 3], ["pens", 2], ["bottles", 7]))
```

该程序将创建以下items2.csv文件。引用项目名称，不引用数字表示的数量。



```bash
"coins",3
"pens",2
"bottles",7
```

## CSV Dialects

尽管CSV格式是一种非常简单的格式，但还是有许多差异，例如不同的定界符，换行或引号字符。因此，有不同的CSV方言可用。

下一个代码示例将打印可用的方言及其特征。



```python
#!/usr/bin/python3

import csv
names = csv.list_dialects()
for name in names:
    print(name)
    dialect = csv.get_dialect(name)
    print(repr(dialect.delimiter), end=" ")
    print(dialect.doublequote, end=" ")
    print(dialect.escapechar, end=" ")
    print(repr(dialect.lineterminator), end=" ")
    print(dialect.quotechar, end=" ")
    print(dialect.quoting, end=" ")
    print(dialect.skipinitialspace, end=" ")
    print(dialect.strict)
```

在csv.list_dialects()返回方言名称的列表和csv.get_dialect()方法返回与方言名称相关联的方言。



```ruby
$ ./dialects.py 
excel
',' 1 None '\r\n' " 0 0 0
excel-tab
'\t' 1 None '\r\n' " 0 0 0
unix
',' 1 None '\n' " 1 0 0
```

程序将打印此输出。有三个内置的方言excel，excel-tab和unix。

## Custom CSV Dialect

在本教程的最后一个示例中，我们将创建一个自定义方言。使用该csv.register_dialect()方法创建自定义方言。



```jsx
#!/usr/bin/python3

import csv
csv.register_dialect("hashes", delimiter="#")
f = open('items3.csv', 'w')
with f:
    writer = csv.writer(f, dialect="hashes")
    writer.writerow(("pencils", 2))
    writer.writerow(("plates", 1))
    writer.writerow(("books", 4))
```

该程序使用（＃）字符作为分隔符。使用方法中的dialect选项指定方言csv.writer()。

该程序将产生以下文件（items3.csv）：



```bash
pencils#2
plates#1
books#4
```

