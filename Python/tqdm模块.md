### tqdm模块是python进度条库, 主要分为两种运行模式

1. 基于迭代对象运行: tqdm(iterator)

```python
import time
from tqdm import tqdm, trange

#trange(i)是tqdm(range(i))的一种简单写法
for i in trange(100):
    time.sleep(0.05)

for i in tqdm(range(100), desc='Processing'):
    time.sleep(0.05)

dic = ['a', 'b', 'c', 'd', 'e']
pbar = tqdm(dic)
for i in pbar:
    pbar.set_description('Processing '+i)
    time.sleep(0.2)
100%|██████████| 100/100 [00:06<00:00, 16.04it/s]
Processing: 100%|██████████| 100/100 [00:06<00:00, 16.05it/s]
Processing e: 100%|██████████| 5/5 [00:01<00:00,  4.69it/s]
```

1. 手动进行更新

```python
import time
from tqdm import tqdm

with tqdm(total=200) as pbar:
    pbar.set_description('Processing:')
    # total表示总的项目, 循环的次数20*10(每次更新数目) = 200(total)
    for i in range(20):
        # 进行动作, 这里是过0.1s
        time.sleep(0.1)
        # 进行进度更新, 这里设置10个
        pbar.update(10)
Processing:: 100%|██████████| 200/200 [00:02<00:00, 91.94it/s]
```

### tqdm模块参数说明

```python
class tqdm(object):
  """
  Decorate an iterable object, returning an iterator which acts exactly
  like the original iterable, but prints a dynamically updating
  progressbar every time a value is requested.
  """

  def __init__(self, iterable=None, desc=None, total=None, leave=False,
               file=sys.stderr, ncols=None, mininterval=0.1,
               maxinterval=10.0, miniters=None, ascii=None,
               disable=False, unit='it', unit_scale=False,
               dynamic_ncols=False, smoothing=0.3, nested=False,
               bar_format=None, initial=0, gui=False):
```

- iterable: 可迭代的对象, 在手动更新时不需要进行设置
- desc: 字符串, 左边进度条描述文字
- total: 总的项目数
- leave: bool值, 迭代完成后是否保留进度条
- file: 输出指向位置, 默认是终端, 一般不需要设置
- ncols: 调整进度条宽度, 默认是根据环境自动调节长度, 如果设置为0, 就没有进度条, 只有输出的信息
- unit: 描述处理项目的文字, 默认是'it', 例如: 100 it/s, 处理照片的话设置为'img' ,则为 100 img/s
- unit_scale: 自动根据国际标准进行项目处理速度单位的换算, 例如 100000 it/s >> 100k it/s

### 下面是实例展示

```python
import time
from tqdm import tqdm

# 发呆0.5s
def action():
    time.sleep(0.5)
with tqdm(total=100000, desc='Example', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
    for i in range(10):
        # 发呆0.5秒
        action()
        # 更新发呆进度
        pbar.update(10000)
Example: 100%|███████████████████████████████████████████████████| 100k/100k [00:05<00:00, 19.6kB/s]
```



