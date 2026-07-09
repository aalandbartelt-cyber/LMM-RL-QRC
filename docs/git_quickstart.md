# Git 日常操作速查

只记这几组命令就够日常协作使用。

## 1. 第一次把仓库克隆到本地

```bash
git clone https://github.com/aalandbartelt-cyber/LMM-RL-QRC.git
cd LMM-RL-QRC
```

## 2. 每次开始写代码前，先拉最新版本

```bash
git pull
```

## 3. 查看自己改了哪些文件

```bash
git status
```

## 4. 本地改完后，加入暂存区

加入全部改动：

```bash
git add .
```

只加入某个文件：

```bash
git add path/to/file.py
```

## 5. 提交到本地 Git

```bash
git commit -m "简短说明你做了什么"
```

例子：

```bash
git commit -m "Add mock policy interface"
```

## 6. 上传到 GitHub

```bash
git push
```

## 7. 最常用完整流程

```bash
git pull
git status
git add .
git commit -m "Update my module"
git push
```

## 8. 注意事项

- 每次开始做事前先 `git pull`。
- 不要上传大文件：视频、模型权重、数据集、训练日志大包。
- 提交信息要写清楚，不要写 `111`、`update`、`test`。
- 如果 `git push` 失败，先 `git pull`，解决冲突后再 push。
- 不确定能不能删别人的文件时，先在群里问。

