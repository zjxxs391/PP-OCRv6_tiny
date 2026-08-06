# -*- coding: utf-8 -*-
"""浏览器端 OCR - 桌面版（tkinter + onnxruntime + PyMuPDF，完全离线）。
用法:
  GUI:    python ocr_app.py            (或打包后的 ocr_app.exe)
  自检:   python ocr_app.py --selftest <图片或PDF> --out <输出txt>
"""
import os
import sys
import json
import math
import threading

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

# ---------- 资源路径（PyInstaller 冻结后从 _MEIPASS 读取打包的数据） ----------
def resource_path(rel):
    base = getattr(sys, '_MEIPASS', os.path.abspath('.'))
    return os.path.join(base, rel)

# ---------- 模型与字符集 ----------
det_session = None
rec_session = None
char_list = []

def load_models():
    global det_session, rec_session, char_list
    with open(resource_path('ppocr_keys_v6_tiny.json'), encoding='utf-8') as f:
        dict_arr = json.load(f)
    char_list = [''] + dict_arr + [' ']
    det_session = ort.InferenceSession(
        resource_path(os.path.join('models', 'PP-OCRv6_det_tiny.onnx')),
        providers=['CPUExecutionProvider'])
    rec_session = ort.InferenceSession(
        resource_path(os.path.join('models', 'PP-OCRv6_rec_tiny.onnx')),
        providers=['CPUExecutionProvider'])

# ---------- PP-OCRv6 参数 ----------
DET_MAX_SIDE = 960
DET_MEAN = [0.485, 0.456, 0.406]
DET_STD = [0.229, 0.224, 0.225]
REC_MEAN = [0.5, 0.5, 0.5]
REC_STD = [0.5, 0.5, 0.5]
REC_HEIGHT = 48

def rgba_to_chw(arr, mean, std):
    h, w = arr.shape[:2]
    img = arr[..., :3].astype(np.float32) / 255.0
    chw = np.empty((3, h, w), dtype=np.float32)
    for c in range(3):
        chw[c] = (img[..., c] - mean[c]) / std[c]
    return chw

def db_boxes(prob, ow, oh, sx, sy):
    thresh, bt, unclip = 0.2, 0.4, 1.4
    bin_flat = (prob > thresh).astype(np.int8).reshape(-1)
    label = np.zeros(ow * oh, dtype=np.int32)
    boxes = []
    for s in range(ow * oh):
        if bin_flat[s] != 1 or label[s] != 0:
            continue
        stack = [s]
        label[s] = -1
        minx, miny, maxx, maxy = ow, oh, 0, 0
        ssum = 0.0
        cnt = 0
        while stack:
            p = stack.pop()
            px, py = p % ow, p // ow
            if px < minx: minx = px
            if px > maxx: maxx = px
            if py < miny: miny = py
            if py > maxy: maxy = py
            ssum += prob.reshape(-1)[p]
            cnt += 1
            if px > 0 and bin_flat[p - 1] and label[p - 1] == 0:
                label[p - 1] = -1; stack.append(p - 1)
            if px < ow - 1 and bin_flat[p + 1] and label[p + 1] == 0:
                label[p + 1] = -1; stack.append(p + 1)
            if py > 0 and bin_flat[p - ow] and label[p - ow] == 0:
                label[p - ow] = -1; stack.append(p - ow)
            if py < oh - 1 and bin_flat[p + ow] and label[p + ow] == 0:
                label[p + ow] = -1; stack.append(p + ow)
        bw, bh = maxx - minx + 1, maxy - miny + 1
        if min(bw, bh) < 3:
            continue
        if ssum / cnt < bt:
            continue
        d = bw * bh * unclip / (2 * (bw + bh))
        boxes.append((max(0, minx - d) * sx, max(0, miny - d) * sy,
                      min(ow, maxx + d) * sx, min(oh, maxy + d) * sy))
    boxes.sort(key=lambda b: (round(b[1] / 5), b[0]))
    return boxes

def ctc_decode(data, T, C):
    text = ''
    prev = -1
    for t in range(T):
        base = t * C
        maxv, idx = -1e9, 0
        for c in range(C):
            v = data[base + c]
            if not math.isfinite(v):
                continue
            if v > maxv:
                maxv, idx = v, c
        if maxv == -1e9:
            continue
        if idx != 0 and idx != prev:
            text += char_list[idx] if idx < len(char_list) else '�'
        prev = idx
    return text.strip()

def ocr_image(arr, progress=None):
    """对单张图像 (H,W,3 numpy) 执行检测 + 识别，返回按阅读顺序排列的文本行。"""
    orig_h, orig_w = arr.shape[:2]
    r = min(1.0, DET_MAX_SIDE / max(orig_w, orig_h))
    det_w = max(32, round(orig_w * r / 32) * 32)
    det_h = max(32, round(orig_h * r / 32) * 32)
    pil = Image.fromarray(arr).resize((det_w, det_h), Image.BILINEAR)
    chw = rgba_to_chw(np.array(pil), DET_MEAN, DET_STD)
    out = det_session.run(None, {det_session.get_inputs()[0].name: chw[None].astype(np.float32)})[0]
    prob = out[0, 0]
    ph, pw = prob.shape
    sx, sy = orig_w / pw, orig_h / ph
    boxes = db_boxes(prob, pw, ph, sx, sy)

    lines = []
    total = len(boxes)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        if progress:
            progress(i + 1, total)
        x0i, y0i = math.floor(x0), math.floor(y0)
        w = min(math.floor(x1 - x0), orig_w - x0i)
        h = min(math.floor(y1 - y0), orig_h - y0i)
        if w < 2 or h < 2:
            continue
        crop = arr[y0i:y0i + h, x0i:x0i + w]
        rec_w = max(8, round(REC_HEIGHT * (x1 - x0) / max(y1 - y0, 1e-6)))
        rec_w = min(rec_w, 2400)
        ri = Image.fromarray(crop).resize((rec_w, REC_HEIGHT), Image.BILINEAR)
        chw2 = rgba_to_chw(np.array(ri), REC_MEAN, REC_STD)
        o = rec_session.run(None, {rec_session.get_inputs()[0].name: chw2[None].astype(np.float32)})[0]
        T, C = o.shape[1], o.shape[2]
        text = ctc_decode(o[0].reshape(-1), T, C)
        if text:
            lines.append(text)
    lines.sort(key=lambda t: 0)  # 已按盒位置排序
    return lines

# ---------- 文件处理 ----------
def is_pdf(path):
    return path.lower().endswith('.pdf')

def process_image(path, progress=None):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert('RGB')
    return ocr_image(np.array(img), progress)

def process_pdf(path, progress=None):
    import fitz
    doc = fitz.open(path)
    n = doc.page_count
    out = []
    for i in range(n):
        if progress:
            progress(i + 1, n)
        pix = doc[i].get_pixmap(dpi=200)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[..., :3]
        page_lines = ocr_image(arr, None)
        out.append(f'===== 第 {i + 1} 页 =====')
        out.extend(page_lines)
    return out

# ---------- 自检模式（供打包后验证，无 GUI） ----------
def selftest():
    # 验证 GUI 预览依赖：PIL.ImageTk 依赖 PIL._tkinter_finder，PyInstaller 必须打包
    try:
        import PIL.ImageTk  # noqa: F401
    except Exception as e:
        print('SELFTEST FAIL: PIL.ImageTk 不可用:', e)
        return 3
    src = None
    out = None
    if '--selftest' in sys.argv:
        src = sys.argv[sys.argv.index('--selftest') + 1]
    if '--out' in sys.argv:
        out = sys.argv[sys.argv.index('--out') + 1]
    load_models()
    if src and os.path.exists(src):
        lines = process_pdf(src) if is_pdf(src) else process_image(src)
        text = '\n'.join(lines)
        if out:
            with open(out, 'w', encoding='utf-8') as f:
                f.write(text)
        return 0
    return 2

# ---------- GUI ----------
def run_gui():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import ImageTk

    root = tk.Tk()
    root.title('浏览器端 OCR - PP-OCRv6')
    root.geometry('760x620')
    root.minsize(620, 480)

    style = ttk.Style(root)
    try:
        style.theme_use('vista')
    except Exception:
        pass

    app = {'file_path': None, 'is_pdf': False, 'busy': False, 'preview': None}

    # ---- 顶部工具栏 ----
    bar = ttk.Frame(root, padding=(10, 8, 10, 4))
    bar.pack(fill='x')
    btn_open = ttk.Button(bar, text='选择图片 / PDF')
    btn_open.pack(side='left')
    btn_ocr = ttk.Button(bar, text='开始识别')
    btn_ocr.pack(side='left', padx=(8, 0))
    lbl_file = ttk.Label(bar, text='未选择文件', anchor='w')
    lbl_file.pack(side='left', padx=(10, 0), fill='x', expand=True)

    # ---- 预览 ----
    preview = tk.Label(root, text='支持 JPG/PNG/BMP/PDF\n图片或 PDF 均可，PDF 将逐页识别',
                       bg='#f1f5f9', fg='#64748b', anchor='center')
    preview.pack(fill='both', expand=False, padx=10, pady=(0, 6))

    # ---- 结果区 ----
    result = tk.Text(root, wrap='char', font=('Microsoft YaHei UI', 10),
                     undo=True, relief='solid', bd=1)
    result.pack(fill='both', expand=True, padx=10, pady=(0, 4))

    # ---- 底部 ----
    foot = ttk.Frame(root, padding=(10, 4, 10, 10))
    foot.pack(fill='x')
    btn_copy = ttk.Button(foot, text='复制结果')
    btn_copy.pack(side='left')
    btn_save = ttk.Button(foot, text='保存为 txt')
    btn_save.pack(side='left', padx=(8, 0))
    lbl_status = ttk.Label(foot, text='就绪', anchor='e')
    lbl_status.pack(side='right', fill='x', expand=True)

    def status(msg):
        lbl_status.config(text=str(msg))

    def open_file():
        path = filedialog.askopenfilename(
            title='选择图片或 PDF',
            filetypes=[('图片 / PDF', '*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff *.pdf'),
                       ('所有文件', '*.*')])
        if not path:
            return
        app['file_path'] = path
        app['is_pdf'] = is_pdf(path)
        lbl_file.config(text=os.path.basename(path))
        status('已选择: ' + os.path.basename(path))
        _show_preview(path, app['is_pdf'])

    def _show_preview(path, pdf):
        try:
            if pdf:
                import fitz
                d = fitz.open(path)
                page = d[0]
                pix = page.get_pixmap(dpi=60)
                img = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
                info = f'PDF · {d.page_count} 页'
                d.close()
            else:
                img = Image.open(path)
                img = ImageOps.exif_transpose(img).convert('RGB')
                info = f'{img.width}×{img.height}'
            img.thumbnail((720, 360))
            photo = ImageTk.PhotoImage(img)
            app['preview'] = photo
            preview.config(image=photo, text='', compound='center', bg='#fff',
                           fg='#94a3b8', height=370)
            lbl_file.config(text=os.path.basename(path) + f'  ({info})')
        except Exception as e:
            preview.config(image='', text='预览失败: ' + str(e))

    def _progress_text(cur, total, what='文本块'):
        return f'识别中 {cur}/{total} {what}…'

    def run_ocr():
        path = app['file_path']
        if not path:
            messagebox.showinfo('提示', '请先选择图片或 PDF 文件')
            return
        if app['busy']:
            return
        if det_session is None:
            status('模型加载中…')
            return
        app['busy'] = True
        btn_ocr.config(state='disabled')
        result.delete('1.0', 'end')
        pdf = app['is_pdf']

        def worker():
            try:
                if pdf:
                    lines = process_pdf(path, lambda c, n: root.after(
                        0, lambda: status(f'PDF 识别中 {c}/{n} 页…')))
                    total = f'共 {len([l for l in lines if l.startswith("=====")])} 页'
                else:
                    lines = process_image(path, lambda c, n: root.after(
                        0, lambda: status(_progress_text(c, n))))
                    total = f'{len(lines)} 行'
                text = '\n'.join(lines)
                root.after(0, lambda: result.insert('1.0', text))
                root.after(0, lambda: status(f'完成 · {total}'))
            except Exception as e:
                root.after(0, lambda: messagebox.showerror('错误', str(e)))
                root.after(0, lambda: status('出错'))
            finally:
                root.after(0, lambda: btn_ocr.config(state='normal'))
                app['busy'] = False

        threading.Thread(target=worker, daemon=True).start()

    def copy_result():
        text = result.get('1.0', 'end').strip()
        if not text:
            return
        root.clipboard_clear()
        root.clipboard_append(text)
        status('已复制 ✓')

    def save_result():
        text = result.get('1.0', 'end').strip()
        if not text:
            messagebox.showinfo('提示', '没有可保存的内容')
            return
        path = filedialog.asksaveasfilename(
            title='保存识别结果', defaultextension='.txt',
            filetypes=[('文本文件', '*.txt')],
            initialfile='ocr_result.txt')
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        status('已保存: ' + os.path.basename(path))

    btn_open.config(command=open_file)
    btn_ocr.config(command=run_ocr)
    btn_copy.config(command=copy_result)
    btn_save.config(command=save_result)

    # 后台加载模型
    def load_worker():
        try:
            root.after(0, lambda: status('模型加载中…'))
            load_models()
            root.after(0, lambda: status('模型已就绪（完全离线）'))
        except Exception as e:
            root.after(0, lambda: messagebox.showerror('模型加载失败', str(e)))
            root.after(0, lambda: status('模型加载失败'))

    threading.Thread(target=load_worker, daemon=True).start()
    root.mainloop()

if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(selftest())
    run_gui()
