# 旧报告怎样跟上代码演进

[返回首页](../README.md) · [验证记录](VALIDATION.md)

报告里的源码 SHA 说明当时运行了什么代码。代码继续修改后，旧报告依然有效，但只描述旧版本。为了同时保留这两件事，证据索引现在可以声明完整的 `source_revision` commit。

[`check_evidence.py`](../scripts/check_evidence.py) 先核对原始报告自身的 SHA 和通过状态，再逐份核对相关源码：

- 索引带 `source_revision`：从本地 Git 对象读取该 commit 的文件。
- 未指定 revision：读取当前工作文件，供新实验使用。
- 显式添加 `--current`：强制拿当前文件比较旧报告；修改过相关代码时，应当失败。

旧报告的 JSON 和其中的 SHA 不变。索引只补充它对应的固定版本；检查通过时会打印实际比较对象。历史匹配不表示当前新版代码已经通过那次硬件测试。

在 Git 检出中运行：

```bash
python scripts/check_evidence.py
python scripts/check_evidence.py --index docs/results/v0.2/evidence.json
python scripts/check_evidence.py --index docs/results/pipeline-benchmark/evidence.json
```

这三份索引分别绑定 v0.1.0、v0.2.0 和首次同步/跨轮基准的 commit。需要审查当前源码是否仍与旧报告完全相同时：

```bash
python scripts/check_evidence.py --index docs/results/pipeline-benchmark/evidence.json --current
```

首轮提交实验已经修改相关执行源码，所以这条命令预期拒绝通过。它与按历史 commit 校验成功并不矛盾。

历史检查要求安装 Git，并已取得指定 commit。新 clone 可按运行指南操作；浅克隆缺少历史时运行 `git fetch --unshallow origin`。脚本不自行联网，也不在找不到历史对象时退回当前文件。源码压缩包没有 Git 历史，需要使用 Git 检出做历史校验；当前文件的检查仍可执行。

CPU 测试同时覆盖：当前文件变化被拒绝、历史匹配后当前文件仍被拒绝、错误 commit 被拒绝、缺失历史对象被拒绝、移动的分支名不允许作为版本，以及原始报告被改写或失败时拒绝。CI 取得完整历史后运行这些索引；新的实机报告另行绑定新源码。
