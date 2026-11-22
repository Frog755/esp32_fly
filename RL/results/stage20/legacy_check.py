#!/usr/bin/env python3
"""legacy 迁移前后一致性检查器（Stage 20）

对每个脚本：在子进程里以 run_name='__legacy_check__' 执行（不触发 if __name__ == '__main__' 主体，
即不训练、不评测、不写文件），工作目录设为临时目录（防止模块级代码写到项目里）。
记录：能否加载成功 / 异常信息 / 模块级字符串变量里所有 /home/ 开头的路径（用于比对迁移前后路径语义是否一致）。
用法：python3 _legacy_check.py out.json <file1> <file2> ...
"""
import sys, os, json, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor

CHILD = r'''
import sys, os, io, json, runpy, contextlib, warnings
warnings.filterwarnings("ignore")
path = sys.argv[1]
sys.argv = [path]
sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
st, err, ns = "ok", "", {}
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        ns = runpy.run_path(path, run_name="__legacy_check__")
except SystemExit as e:
    st, err = "exit", repr(e)[:300]
except BaseException as e:
    st, err = "fail", (type(e).__name__ + ": " + str(e))[:300]
paths = {k: v for k, v in ns.items() if isinstance(v, str) and v.startswith("/home/")
         and not k.startswith(("_LEGACY", "_lg", "__"))}
mods = sorted(m for m, o in sys.modules.items() if getattr(o, "__file__", None) and "/esp32_fly/RL/" in (o.__file__ or ""))
print("@@" + json.dumps(dict(status=st, err=err, paths=paths, local_mods=mods), ensure_ascii=False))
'''


def run(path):
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, PYTHONWARNINGS='ignore', MUJOCO_GL='egl', PYTHONDONTWRITEBYTECODE='1')
        try:
            p = subprocess.run([sys.executable, '-c', CHILD, os.path.abspath(path)], cwd=td, env=env,
                               capture_output=True, text=True, timeout=180)
            line = [l for l in p.stdout.splitlines() if l.startswith('@@')]
            if line:
                r = json.loads(line[-1][2:])
            else:
                r = dict(status='crash', err=(p.stderr or p.stdout)[-300:], paths={}, local_mods=[])
        except subprocess.TimeoutExpired:
            r = dict(status='timeout', err='>180s', paths={}, local_mods=[])
    r['local_mods'] = [os.path.basename(m) for m in r['local_mods']]
    return path, r


if __name__ == '__main__':
    out, files = sys.argv[1], sys.argv[2:]
    with ThreadPoolExecutor(8) as ex:
        res = dict(ex.map(run, files))
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
    for f, r in res.items():
        print(f"{r['status']:<7} {f}  {r['err'][:120]}")
