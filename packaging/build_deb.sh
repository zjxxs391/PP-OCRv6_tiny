#!/usr/bin/env bash
# 组装 .deb 安装包
# 用法: build_deb.sh <OCR-linux 二进制> <icon.png> <版本> <输出 .deb 路径>
set -e

BIN="$1"
ICON="$2"
VERSION="$3"
OUT="$4"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG=/tmp/debpkg

rm -rf "$PKG"
mkdir -p "$PKG/DEBIAN" \
         "$PKG/usr/bin" \
         "$PKG/usr/share/applications" \
         "$PKG/usr/share/icons/hicolor/256x256/apps"

# 二进制安装到 /usr/bin/ocr
cp "$BIN" "$PKG/usr/bin/ocr"
chmod 755 "$PKG/usr/bin/ocr"

# 图标
cp "$ICON" "$PKG/usr/share/icons/hicolor/256x256/apps/ocr.png"

# 桌面条目（安装到系统应用目录，供开始菜单/桌面集成使用）
cp "$SCRIPT_DIR/ocr.desktop" "$PKG/usr/share/applications/ocr.desktop"

# control
cat > "$PKG/DEBIAN/control" <<EOF
Package: ocr
Version: $VERSION
Section: utils
Priority: optional
Architecture: amd64
Maintainer: zjxxs391 <zjxxs391@users.noreply.github.com>
Depends: libc6, libx11-6
Description: PP-OCRv6 offline OCR desktop app
 Image and PDF OCR, fully offline, CPU inference.
Homepage: https://github.com/zjxxs391/PP-OCRv6_tiny
EOF

# 安装/卸载脚本（postinst 在桌面创建快捷方式）
cp "$SCRIPT_DIR/postinst" "$PKG/DEBIAN/postinst"
cp "$SCRIPT_DIR/prerm" "$PKG/DEBIAN/prerm"
chmod 755 "$PKG/DEBIAN/postinst" "$PKG/DEBIAN/prerm"

dpkg-deb --build "$PKG" "$OUT"
echo "已生成 .deb: $OUT"
