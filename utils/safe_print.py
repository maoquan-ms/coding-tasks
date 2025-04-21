import sys

# --------------------------- 打印工具 --------------------------- #
def safe_print(*args, **kwargs):
    r"""safe print for non-UTF8 encodings.
    This function ensures that the output is encoded correctly"""
    enc = (sys.stdout.encoding or "").lower()
    if enc.startswith("utf"):
        print(*args, **kwargs)
    else:
        text = " ".join(map(str, args))
        print(text.encode(enc or "gbk", errors="replace").decode(enc or "gbk"), **kwargs)