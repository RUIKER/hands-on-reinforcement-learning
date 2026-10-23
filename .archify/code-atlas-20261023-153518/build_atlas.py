#!/usr/bin/env python3
"""把各目录的代码块 JSON 汇总成 Archify explainer 草稿，并做 AST 覆盖率检查。"""
from __future__ import annotations

import ast
import json
import os
import re
import sys

ROOT = "/Users/yanrui/Downloads/硕士/gpulab/Reinforcement Learning/code"
BASE = os.path.join(ROOT, ".archify", "code-atlas-20261023-153518")
DATA = os.path.join(BASE, "data")

META = {
    "01": ("01_Cliff Walking", "价值型", "`train.py`"),
    "02": ("02_DQN", "价值型", "`practice.py`"),
    "03": ("03_Double DQN、Dueling DQN", "价值型", "`main.py`"),
    "04": ("04_策略梯度算法_Reinforce", "策略梯度", "`main.py`"),
    "05": ("05_Mountain Car", "策略梯度", "`main.py`"),
    "06": ("06_Mountain Car PyTorch", "策略梯度", "`main.py`"),
    "07": ("07_TRPO", "策略梯度", "`*/main.py`"),
    "08": ("08_PPO", "策略梯度", "`*/main.py`"),
    "09": ("09_DDPG", "连续控制", "`*/main.py`"),
    "10": ("10_TD3", "连续控制", "`Pendulum/main.py`"),
    "11": ("11_SAC", "连续控制", "`Pendulum/main.py`"),
}

TITLES = {
    "01": "01_Cliff Walking：表格型方法如何落到文件？",
    "02": "02_DQN：最小 DQN 练习由哪些块组成？",
    "03": "03_Double DQN、Dueling DQN：三种价值变体如何共用 pipeline？",
    "04": "04_策略梯度算法_Reinforce：REINFORCE 的文件分工？",
    "05": "05_Mountain Car：线性 Actor-Critic 如何拆文件？",
    "06": "06_Mountain Car PyTorch：MLP 版本改了哪些文件？",
    "07": "07_TRPO：离散与连续两套 TRPO 如何组织？",
    "08": "08_PPO：离散与连续两套 PPO 如何组织？",
    "09": "09_DDPG：标准版与 CartPole 教学版如何组织？",
    "10": "10_TD3：双 critic 与延迟更新落在哪里？",
    "11": "11_SAC：最大熵与自动温度落在哪里？",
}


def load() -> dict[str, dict]:
    out = {}
    for rid in META:
        path = os.path.join(DATA, f"{rid}.json")
        if not os.path.exists(path):
            print(f"MISSING DATA: {path}")
            continue
        with open(path, encoding="utf-8") as fh:
            out[rid] = json.load(fh)
    return out


def norm_path(p: str) -> str:
    p = p.replace("\\", "/")
    if p.startswith(ROOT):
        p = os.path.relpath(p, ROOT)
    p = p.lstrip("./")
    return p


def ast_names(path: str) -> set[str]:
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(f"{node.name}.{sub.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def json_names(file_entry: dict) -> set[str]:
    names: set[str] = set()
    for block in file_entry.get("blocks", []):
        kind = block.get("kind", "")
        name = (block.get("name") or "").strip()
        if kind in {"类", "方法", "函数"} and name:
            names.add(name)
    return names


def check(data: dict[str, dict]) -> list[str]:
    problems: list[str] = []
    for rid, payload in sorted(data.items()):
        root_dir = META[rid][0]
        disk = set()
        for base, dirs, files in os.walk(os.path.join(ROOT, root_dir)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in files:
                if fn.endswith(".py"):
                    disk.add(norm_path(os.path.join(base, fn)))
        got = {norm_path(f.get("path", "")) for f in payload.get("files", [])}
        for p in sorted(disk - got):
            problems.append(f"[{rid}] JSON 缺少文件: {p}")
        for p in sorted(got - disk):
            problems.append(f"[{rid}] JSON 多出文件: {p}")
        for f in payload.get("files", []):
            p = norm_path(f.get("path", ""))
            full = os.path.join(ROOT, p)
            if not os.path.exists(full):
                continue
            try:
                expected = ast_names(full)
            except SyntaxError as exc:
                problems.append(f"[{rid}] AST 解析失败 {p}: {exc}")
                continue
            actual = json_names(f)
            for n in sorted(expected - actual):
                problems.append(f"[{rid}] {p} 缺少代码块: {n}")
            for n in sorted(actual - expected):
                problems.append(f"[{rid}] {p} 多出代码块: {n}")
    return problems


def clean(text: str) -> str:
    text = (text or "").replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(">=", "≥").replace("<=", "≤")
    text = text.replace("*", "\\*")
    text = text.replace("|", "\\|")
    return text


def block_rows(file_entry: dict, seq: list[int]) -> list[str]:
    rows = ["| 类型 | 代码块 | 行 | 逻辑 |", "| --- | --- | --- | --- |"]
    for b in file_entry.get("blocks", []):
        kind = clean(b.get("kind", ""))
        name = clean(b.get("name", ""))
        line = clean(str(b.get("line", "")))
        logic = clean(b.get("logic", ""))
        shown = name if kind in {"导入与常量", "入口"} else f"`{name}`"
        seq[0] += 1
        rows.append(f"| {kind} {{#blk{seq[0]:04d}}} | {shown} | {line} | {logic} |")
    return rows


def build(data: dict[str, dict]) -> str:
    total_files = sum(len(p.get("files", [])) for p in data.values())
    total_blocks = sum(
        len(f.get("blocks", [])) for p in data.values() for f in p.get("files", [])
    )
    lines: list[str] = [
        "# 强化学习代码地图：每个文件与每个代码块 {#code-atlas}",
        "",
        f"这份代码地图覆盖 11 个算法目录、{total_files} 个 Python 文件和 {total_blocks} 个代码块条目；每个文件按导入与常量、类、方法、函数、入口拆开，逐块说明输入、输出和关键步骤。",
        "",
        "- 01–03 是价值型方法：Q 表、DQN、Double DQN 与 Dueling DQN。",
        "- 04–08 是策略梯度方法：REINFORCE、Actor-Critic、TRPO 与 PPO。",
        "- 09–11 是连续控制方法：DDPG、TD3 与 SAC，均使用 replay buffer。",
        "- 03–11 共用 config、environment、agent、networks、trainer、main 的拆分方式；01 与 02 是简化结构。",
        "",
        '@answer scope="11 个算法目录 · 当前工作区文件 · 行号为当前文件行号"',
        "@meta locale=zh-CN quality_profile=showcase",
        "",
        "## 代码库如何组织？ {#overview span=12}",
        "",
        "@facts id=atlas-facts",
        "| 算法目录 | 11 |",
        f"| Python 文件 | {total_files} |",
        f"| 代码块条目 | {total_blocks} |",
        "| 阅读分支 | 3 |",
        "",
        '@tree id=repo-tree title="01–11 目录结构"',
        "| root | 强化学习代码库 | - |",
    ]
    for rid, (root_dir, branch, _entry) in META.items():
        lines.append(f"| a{rid} | {root_dir} | root |")
        if rid in {"07", "08", "09"}:
            lines.append(f"| a{rid}_cart | CartPole | a{rid} |")
            lines.append(f"| a{rid}_pend | Pendulum | a{rid} |")
        elif rid in {"10", "11"}:
            lines.append(f"| a{rid}_pend | Pendulum | a{rid} |")
    lines += [
        "",
        "| 目录 | 分支 | 入口 |",
        "| --- | --- | --- |",
    ]
    for rid, (root_dir, branch, entry) in META.items():
        lines.append(f"| `{root_dir}` | {branch} | {entry} |")
    lines += [
        "",
        '@callout level=note title="读取范围"',
        "按 2026-10-23 的当前工作区读取；01–11 的重命名与文档尚未提交，行号只对应当前文件，不对应 Git HEAD。",
        "",
    ]

    seq = [0]
    for rid, (root_dir, branch, entry) in META.items():
        payload = data.get(rid)
        if not payload:
            continue
        files = payload.get("files", [])
        nblocks = sum(len(f.get("blocks", [])) for f in files)
        lines += [
            f"## {TITLES[rid]} {{#algo{rid} span=12}}",
            "",
            f"{clean(payload.get('summary', ''))} 本目录共 {len(files)} 个 Python 文件、{nblocks} 个代码块条目。",
            "",
        ]
        for f in files:
            path = norm_path(f.get("path", ""))
            purpose = clean(f.get("purpose", ""))
            flines = f.get("lines", "")
            lines += [
                f"**`{path}`** · {flines} 行 · {purpose}",
                "",
                *block_rows(f, seq),
                "",
            ]
    return "\n".join(lines) + "\n"


def main() -> int:
    data = load()
    problems = check(data)
    if problems:
        print(f"覆盖率问题 {len(problems)} 条：")
        for p in problems[:200]:
            print(" -", p)
        if len(problems) > 200:
            print(f" ... 其余 {len(problems) - 200} 条省略")
    else:
        print("AST 覆盖率检查通过：每个文件的类、方法、顶层函数都在 JSON 中。")
    if not data:
        return 2
    draft_path = os.path.join(BASE, "candidate.archify.md")
    with open(draft_path, "w", encoding="utf-8") as fh:
        fh.write(build(data))
    print("草稿:", draft_path)
    print("文件数:", sum(len(p.get("files", [])) for p in data.values()))
    print("代码块:", sum(len(f.get("blocks", [])) for p in data.values() for f in p.get("files", [])))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
