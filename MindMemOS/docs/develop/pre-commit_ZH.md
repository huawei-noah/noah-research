# Pre-commit 检查

本仓库使用 `pre-commit` 在提交前自动检查代码格式和密钥泄露。

## 安装

先安装 `gitleaks`。

macOS:

```bash
brew install gitleaks
```

Windows:

```powershell
winget install gitleaks
gitleaks version
```

也可以用 Chocolatey 或 Scoop：

```powershell
choco install gitleaks
scoop install gitleaks
```

Linux:

```bash
curl -sSfL https://raw.githubusercontent.com/gitleaks/gitleaks/master/install.sh | sh -s -- -b /usr/local/bin
gitleaks version
```

然后在仓库根目录安装依赖和 Git hooks：

```bash
make dev-setup
```

如果已经装过开发依赖，只想安装 hooks：

```bash
make hooks-install
```

## 使用

正常提交即可：

```bash
git add .
git commit -m "your message"
```

提交前会自动运行：

- `ruff check --fix`：修复可自动处理的 lint / import 顺序问题
- `ruff format`：统一 Python 代码格式
- `gitleaks protect --staged --redact`：扫描 staged 内容里的密钥

也可以手动检查已暂存文件：

```bash
uv run pre-commit run
```

## 常见情况

如果 Ruff 修改了文件，重新 `git add` 后再提交。

如果 Gitleaks 报密钥泄露，不要提交；先删除或替换为测试占位值。如果是误报，再考虑加白名单。
