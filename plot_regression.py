# plot_regression.py : Chainer Trainer Extension for plotting with regression problems
# Author: hiromi-mi
# License: Unlicense or CC0 (same as public domain.)

import os.path
import chainer
import chainer.training as training
import chainer.backends.cuda as cuda
import matplotlib.pyplot as plt

def plot_regression(model, ideal, x_train, t_train, x_test, filename="figure_{epoch}.png"):

    """ Trainer 上の1次元関数プロット機構

    引数:
    - model : chainer.Model : CPU/GPU 両対応
    - ideal : numpy.ndarray を引数にとる関数
    - x_train, t_train : train データの対応を表す numpy.ndarray
      Chainer の仕様に合わせて、[[-1], [2], [4], [5], ...] という形のデータでも
      Matplotlib numpy.linspace の仕様に合わせて [-1, 2, 3, 4]... という形のデータでも対応
    - x_test : テストデータを表す numpy.ndarray
      Chainer の仕様に合わせて、[[-1], [2], [4], [5], ...] という形のデータのみ対応
    - filename : プロットされた画像の名前
                 trainer の引数 out として指定されたディレクトリに格納される
                 既定では "result/figure_{epoch:03}.png"
                 なお、{epoch} とするとそこにepoch 番号が入り、
                 "result/figure_023.png" などと保存される
                 また、{:03} などとつけると、1桁や2桁の数値は3桁になるよう0で埋められる

    例: MLP に三角関数を学習させてみる

    仮定:
    # args.device はデバイス番号を表す *文字列*
    # args.batchsize はミニバッチの個数を表す数値
    # args.epoch はEpoch 数を表す数値
    # args.out は出力先
    # MLP() はモデル
    # Regression() は回帰関数を表すメタモデル

    import chainer
    import chainer.links as L
    from chainer import training
    from chainer.training import extensions
    import numpy as np

    import plot_regression

    # 引数として --device=1 などが指定されたとしている
    device = chainer.get_device(args.device)
    device.use() # device option の値に応じて GPU などを利用する

    # MLP というモデルが定義されているとしている
    model = MLP()
    model.to_device(device) # GPU などに変換

    # 損失関数を計算するためのモデル
    model_with_loss = Regression(model)
    model_with_loss.to_device(device)

    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(model_with_loss)

    # データセットの準備
    ideal = lambda x: np.sin(x) # (何かしら) x が渡されると np.sin(x) を返す関数
    x_train = np.linspace(-np.pi, np.pi, 10, dtype=np.float32)
    # -3.14 から 3.14 まで10段階
    x_train = x_train.reshape(-1, 1)
    # Chainer の仕様上、[[-1], [2]] などという形式のデータセットしか受け付けない
    # のでそのように変換
    t_train = ideal(x_train)
    train = chainer.datasets.TupleDataset(x_train, t_train)

    # テストデータの準備
    x_test = np.linspace(-np.pi, np.pi, 37, dtype=np.float32)
    # -3.14 から 3.14 まで37段階
    # 37段階にすると丁度 train sample の間に3つ test sample が配置される構図になる
    x_test = x_test.reshape(-1, 1)
    # Chainer の仕様では、[[-1], [2]] などという形式のデータセットしか受け付けない
    # のでそのように変換
    t_test = ideal(x_test)
    test = chainer.datasets.TupleDataset(x_test, t_test)

    # Trainer の準備
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model_with_loss, device=device))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    # この関数 plot_regression を Extension として使う
    trainer.extend(plot_regression.plot_regression(model, ideal, x_train, t_train, x_test),
                   trigger=(10, 'epoch'))
    # trigger 引数は必ずしも必要ない. 例えば 10epoch ごとにグラフを描画するときに使う

    # 学習実行
    trainer.run()
    """

    # 現状のデバイス (CPU/GPU/iDeep/ChainerX) が何かを、model から特定する
    # xp = backend.get_array_module(model) # numpy (np) または cupy
    device = chainer.backend.get_device_from_array(model)
    # デバイスを表すDictionary

    # データ形式を (-1, 1) から (-1,) に変換
    # Chainer の仕様では [[-1], [2], [3], ...] などという形式のみ対応
    # matplotlib の仕様では [-1, 2, 3, ...] などという形式のみ対応
    x_test_onedimensional = x_test.reshape(-1)
    x_train_onedimensional = x_train.reshape(-1)
    t_train_onedimensional = t_train.reshape(-1)

    # 渡されたものを model() に適用できる CPU/GPU/iDeep などに変換
    chainer.Variable(x_test).to_device(device)

    # デコレータ (関数をとり、何かしらの処理をつけくわえて、関数を返す)
    @training.make_extension(trigger=(1, 'epoch'))
    def _plot_regression_internal(trainer):
        # この計算結果が学習処理に反映されないように (gradient が加算されないように)
        with chainer.using_config('train', False):
            y_test = cuda.to_cpu(model(x_test).data)
        # .data をつけることで numpy or cupy 形式に変換し、 cuda.to_cpu() を通すと numpy 形式に変換

        # テストデータを用意
        t_test = ideal(x_test)

        # データ形式を変換
        # Chainer の仕様では [[-1], [2], [3], ...] などという形式のみ対応
        # matplotlib の仕様では [-1, 2, 3, ...] などという形式のみ対応
        y_test = y_test.reshape(-1)
        t_test = t_test.reshape(-1)

        plt.plot(x_test_onedimensional, y_test, label="y: model's output")
        plt.plot(x_test_onedimensional, t_test, label="t: target functions' output")
        # marker というのは表示形式のこと. こうすると点で打たれる
        # . , o v ^ < などがある. - とすると直線になる
        plt.plot(x_train_onedimensional, t_train_onedimensional, label="trained points", marker=".")

        # x軸とy 軸にラベルをつける
        plt.xlabel("$x$") # $ をつけると TeX 数式が利用できる
        plt.xlabel("$y$") # $ をつけると TeX 数式が利用できる

        # epoch 番号を表題にする
        plt.title("On Epoch {}".format(trainer.updater.epoch))
        plt.legend() # plot にて指定された内容にて凡例を表示する

        # グリッドを表示 (b=True とすると表示)
        plt.grid(b=True)

        # ファイルパス
        # trainer.out に Trainer() にて指定した保存先ディレクトリがある
        # trainer.updater.epoch によりepoch がわかる
        # {epoch:03} を format() 関数で実際のepoch に置き換えてディレクトリ名と結合
        filepath = os.path.join(trainer.out, filename.format(epoch=trainer.updater.epoch))
        plt.savefig(fname=filepath)

        # 後片付け
        plt.close() # plt.cla + plt.clf を両方行なってくれる
        # TODO 本来は Figure と Axis の概念を説明すべき

    return _plot_regression_internal
