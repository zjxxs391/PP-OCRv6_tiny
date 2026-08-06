# OCR 桌面版（Debian 11 构建说明）

PP-OCRv6 离线 OCR 桌面程序：支持图片与 PDF，中文/英文识别，完全本地推理。

## 用 GitHub Actions 构建（推荐）

仓库已包含 `.github/workflows/build-debian11.yml`：

1. 把本仓库推到 GitHub（或手动触发 Actions）
2. Actions 会自动在 **Debian 11 容器**内执行 `Dockerfile.debian11` 构建
3. 构建完成后自动运行内置自检（识别测试图），然后**发布到 Releases**（版本号 = 编译成功日期，如 `2026.08.06`）
4. Release 包含两个资产：`OCR-linux`（直接可执行文件）和 `OCR-<日期>-x86_64.AppImage`（便携格式）
5. 也可在 Actions 页面 → 对应 run → **Artifacts** 下载

> 为什么必须在 Debian 11 容器里构建：Linux 可执行文件链接构建机的 glibc，
> 在较新系统上构建的二进制无法在 Debian 11（glibc 2.31）上运行。

## 在本机 Debian 11 上直接构建

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-tk python3-dev
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-linux.txt
python3 -m PyInstaller --onefile --windowed --name OCR-linux \
  --icon ocr.ico \
  --add-data "models:models" \
  --add-data "ppocr_keys_v6_tiny.json:." \
  ocr_app.py
# 自检
./dist/OCR-linux --selftest build_assets/test.png --out selftest.txt
```

产出：`dist/OCR-linux`

## 使用

- **GUI**：双击 `OCR-linux`（或 `.AppImage`），选择图片或 PDF → 开始识别 → 复制 / 保存 txt
- **命令行自检**：`./OCR-linux --selftest <图片或PDF> --out <输出txt>`

AppImage 便携版跨发行版通用；若系统未装 FUSE，运行需加参数：
`./OCR-2026.08.06-x86_64.AppImage --appimage-extract-and-run`

## 目标机器运行要求

- 桌面环境（tkinter GUI 需要 X11/Wayland）
- 中文显示需要字体：`sudo apt-get install -y fonts-noto-cjk`
- 完全离线，无需联网

## 源码结构

```
ocr_app.py                 # 桌面程序（tkinter + onnxruntime + PyMuPDF）
models/                    # PP-OCRv6 det/rec ONNX 模型
ppocr_keys_v6_tiny.json    # 6904 字符集
requirements-linux.txt     # Debian 11 / Python 3.9 兼容的依赖版本
Dockerfile.debian11        # Debian 11 构建容器
build_assets/test.png      # 构建自检用测试图
```
