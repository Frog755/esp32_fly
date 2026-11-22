#!/usr/bin/env python3
"""Stage 20 迁移：把历史脚本 git mv 到 RL/legacy/<组>/，并给每个 .py 注入路径引导块。

引导块做两件事：
  1. _LEGACY_ORIG_DIR = 文件原来所在目录；把源码里的 os.path.dirname(os.path.abspath(__file__)) /
     os.path.dirname(__file__) 替换为它 → 原有 ../env、../models、../results 等相对路径语义完全不变；
  2. 把 RL/、RL/env、RL/scripts 与每个 legacy 子目录加进 sys.path → 跨目录的旧导入照常可用。
除此之外不改动任何逻辑。
"""
import ast, os, re, subprocess, sys

RL = '/home/frog/embodied_ai/esp32_fly/RL'
os.chdir(RL)
MOVE = {
    'rl_v0_ppo': ['train_ppo.py', 'eval.py', 'play.py', 'eval_phase.py', 'eval_hierarchical.py', 'eval_by_height.py',
                  'report.py', 'compare_tb.py', 'monitor_training.py', 'export_policy_c.py',
                  'env/quad_sim.py', 'env/quad_sim_fast_v2_v28_backup.py'],   # quad_sim_fixed 被主线 quad_sim_fast_v2 懒加载，留在 env/
    'rl_vec': ['scripts/train_planA.py', 'scripts/train_sac.py', 'scripts/train_residual.py', 'scripts/train_residual_anchored.py',
               'scripts/train_residual_omegafirst.py', 'scripts/train_rate.py', 'scripts/train_rate_anchored.py',
               'scripts/train_expert_sil.py', 'scripts/collect_anchor_data.py', 'scripts/collect_anchor_data_expert.py',
               'scripts/eval_residual.py', 'scripts/eval_planA_v2.py', 'scripts/export_tb_plots.py',
               'env/vec_residual_env.py', 'env/vec_rate_env.py', 'env/vec_expert_residual_env.py'],
    'flip_table': ['scripts/flip_lto.py', 'scripts/flip_lto3d.py', 'scripts/flip_table.py', 'scripts/flip_table2.py',
                   'scripts/build_flip_table3d.py', 'scripts/export_table_c.py', 'scripts/eval_table_vs_v5.py'],
    'raptor': ['scripts/raptor_ref.py', 'scripts/raptor_policy_np.py', 'scripts/test_raptor_in_oursim.py'],
    'diagnose': ['scripts/diagnose.py', 'scripts/diagnose_e2e_on_handoff.py', 'scripts/diagnose_handoff_split.py',
                 'scripts/diagnose_handoff_timing.py', 'scripts/diagnose_omega_curve.py', 'scripts/diagnose_pd_sweep.py',
                 'scripts/baseline_recovery.py', 'eval_pd_e2e.py'],
}
EXTRA = {'run_baseline.sh': 'legacy/rl_v0_ppo/run_baseline.sh', 'README.md': 'legacy/OLD_RL_README.md'}

# 模块名必须全局唯一（迁移后它们都会在 sys.path 上）
import glob
allpy = glob.glob('*.py') + glob.glob('env/*.py') + glob.glob('scripts/*.py')
names = [os.path.basename(f) for f in allpy]
dup = {n for n in names if names.count(n) > 1}
assert not dup, f'模块名重复: {dup}'

BOOT = '''# --- [Stage 20 legacy 迁移引导] 本文件原位于 RL/{orig_disp}，现移至 RL/legacy/{group}/ ---
# _LEGACY_ORIG_DIR 指回原目录：原有基于 __file__ 的相对路径（../env、../models 等）语义不变；
# 并把主线 env/、scripts/ 与各 legacy 子目录加入 sys.path，跨目录的旧导入照常可用。逻辑未改动。
import os as _lg_os, sys as _lg_sys
_LEGACY_RL = _lg_os.path.dirname(_lg_os.path.dirname(_lg_os.path.dirname(_lg_os.path.abspath(__file__))))
_LEGACY_ORIG_DIR = {orig_expr}
for _lg_p in [_LEGACY_RL, _lg_os.path.join(_LEGACY_RL, 'env'), _lg_os.path.join(_LEGACY_RL, 'scripts')] + sorted(
        _lg_os.path.join(_LEGACY_RL, 'legacy', _lg_d) for _lg_d in _lg_os.listdir(_lg_os.path.join(_LEGACY_RL, 'legacy'))
        if _lg_os.path.isdir(_lg_os.path.join(_LEGACY_RL, 'legacy', _lg_d)) and not _lg_d.startswith(('.', '_'))):
    if _lg_p not in _lg_sys.path:
        _lg_sys.path.insert(1, _lg_p)
# --- [/Stage 20 legacy 迁移引导] ---
'''


def git(*a):
    r = subprocess.run(['git', *a], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def inject(path, orig_dir, group):
    src = open(path, encoding='utf-8').read()
    n1 = src.count('os.path.dirname(os.path.abspath(__file__))')
    s = src.replace('os.path.dirname(os.path.abspath(__file__))', '_LEGACY_ORIG_DIR')
    n2 = s.count('os.path.dirname(__file__)')
    s = s.replace('os.path.dirname(__file__)', '_LEGACY_ORIG_DIR')
    assert '__file__' not in s, f'{path}: 仍有未处理的 __file__ 用法'
    tree = ast.parse(s)
    body = tree.body; i = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant) and isinstance(body[0].value.value, str):
        i = 1
    while i < len(body) and isinstance(body[i], ast.ImportFrom) and body[i].module == '__future__':
        i += 1
    assert i < len(body)
    node = body[i]
    line0 = (node.decorator_list[0].lineno if getattr(node, 'decorator_list', None) else node.lineno) - 1
    lines = s.splitlines(keepends=True)
    orig_expr = f"_lg_os.path.join(_LEGACY_RL, {orig_dir!r})" if orig_dir else '_LEGACY_RL'
    boot = BOOT.format(orig_disp=(orig_dir + '/' if orig_dir else ''), group=group, orig_expr=orig_expr)
    new = ''.join(lines[:line0]) + boot + ''.join(lines[line0:])
    ast.parse(new)
    open(path, 'w', encoding='utf-8').write(new)
    return n1, n2


os.makedirs('legacy', exist_ok=True)
moved = []
for g, fs in MOVE.items():
    os.makedirs(f'legacy/{g}', exist_ok=True)
    for f in fs:
        dst = f'legacy/{g}/{os.path.basename(f)}'
        git('mv', f, dst)
        n1, n2 = inject(dst, os.path.dirname(f), g)
        moved.append((f, dst, n1, n2))
for f, dst in EXTRA.items():
    git('mv', f, dst)

# run_baseline.sh：根目录改为向上两级、训练入口改为新路径
p = 'legacy/rl_v0_ppo/run_baseline.sh'
s = open(p, encoding='utf-8').read()
a = 'RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"'
assert s.count(a) == 1
s = s.replace(a, 'RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # Stage 20：脚本已移到 RL/legacy/rl_v0_ppo/')
assert s.count('python3 train_ppo.py') == 1
s = s.replace('python3 train_ppo.py', 'python3 legacy/rl_v0_ppo/train_ppo.py')
open(p, 'w', encoding='utf-8').write(s)

for f, dst, n1, n2 in moved:
    print(f'{f:<40} -> {dst:<44} 替换 dirname(abspath)×{n1} dirname×{n2}')
print('迁移完成：', len(moved), '个 .py +', len(EXTRA), '个其他文件')
