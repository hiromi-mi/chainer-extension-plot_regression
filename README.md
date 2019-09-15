# Chainer Trainer Extension to Plot Regresson Problem

Chainer の Trainer Extension のひとつ.

## 依存関係

* Python 3
* Chainer
* matplotlib

## 使い方

ソースコード例は次の通り

```python
import numpy
import plot_regression
model = Model()
ideal = lambda x: numpy.sin(x)
x_train = np.asarray([-1,1,2]).astype(np.float32).reshape(-1, 1)
t_train = ideal(x_train)
x_test = np.asarray([-1,0,1,2]).astype(np.float32).reshape(-1, 1)
t_test = ideal(x_test)
trainer.extend(plot_regression.plot_regression(model, ideal, x_train, t_train, x_test),
                   trigger=(10, 'epoch'))
```

詳細な使い方は `plot_regression` 先頭のドキュメントを参照。

## ライセンス

[The Unlicense](LICENSE)
