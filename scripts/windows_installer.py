"""个人免管理员安装器；数据始终留在 LOCALAPPDATA/AegisCode。"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path


def install(target: Path):
    target = target.resolve()
    if target.exists() and any(target.iterdir()):
        raise ValueError("请选择空目录；升级请先退出旧服务并选择新的安装目录")
    payload = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "payload"
    archive_path = payload / "AegisCode-portable.zip"
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            relative = Path(entry.filename)
            if not relative.parts or relative.parts[0] != "AegisCode":
                raise ValueError("安装包目录不合法")
            destination = (target / Path(*relative.parts[1:])).resolve()
            if not destination.is_relative_to(target):
                raise ValueError("安装包路径越界")
            if entry.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
    return target / "AegisCode.exe"


def main():
    parser = argparse.ArgumentParser(description="AegisCode 安装器")
    parser.add_argument("--target", type=Path, help="非交互安装到空目录")
    arguments = parser.parse_args()
    if arguments.target:
        install(arguments.target)
        return
    import tkinter as tk
    from tkinter import filedialog, messagebox

    window = tk.Tk()
    window.title("安装 AegisCode")
    window.geometry("550x240")
    path = tk.StringVar(value=str(Path(os.environ["LOCALAPPDATA"]) / "Programs/AegisCode"))
    tk.Label(window, text="AegisCode · 私有任务工作台", font=("Microsoft YaHei", 16)).pack(pady=18)
    tk.Label(
        window,
        text=(
            "安装不需要管理员权限。模型需要在数据目录 .env 中配置。\n"
            "升级和卸载保留独立的用户数据目录。"
        ),
    ).pack()
    tk.Entry(window, textvariable=path, width=65).pack(pady=10)

    def select():
        chosen = filedialog.askdirectory()
        if chosen:
            path.set(chosen)

    def submit():
        try:
            executable = install(Path(path.get()))
            messagebox.showinfo("安装完成", f"双击以下文件启动：\n{executable}")
            window.destroy()
        except Exception as error:
            messagebox.showerror("安装失败", str(error))

    tk.Button(window, text="选择目录", command=select).pack(side="left", padx=80)
    tk.Button(window, text="安装", command=submit, width=15).pack(side="right", padx=80)
    window.mainloop()


if __name__ == "__main__":
    main()
