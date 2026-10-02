#!/usr/bin/env python3
"""Station Harness v2 — 执行验证型 Agent 评测。

固定 agent loop：所有被测模型走同一套工具（read/write/edit/bash）与同一系统提示。
每个任务在独立临时目录执行，verify.sh 判定。支持 pass@k / pass^k 双口径 + Wilson CI。

用法：
    python harness/run_eval.py --models deepseek-flash,glm-5.3-flash --tasks tasks/ --repeats 3
    python harness/run_eval.py --list-tasks          # 只列任务
    python harness/run_eval.py --validate-tasks      # verifier 防抖检查（参考解跑3遍）
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time, urllib.request, urllib.error
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR.parent / "src"))
try:
    from llm_eval.stats import pass_at_k, wilson_ci  # 若 v1 里有则复用
except Exception:
    import math
    def pass_at_k(results, k):
        n, c = len(results), sum(results)
        if n - c < k: return 1.0
        return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))
    def wilson_ci(p, n, z=1.96):
        if n == 0: return (0.0, 0.0)
        d = 1 + z*z/n
        c = (p + z*z/(2*n)) / d
        h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
        return (max(0.0, c-h), min(1.0, c+h))

SYSTEM_PROMPT = """你是一个在沙箱里工作的软件工程 agent。你可以使用以下工具：

read(path) — 读取工作目录内文件
write(path, content) — 写文件（覆盖）
edit(path, old, new) — 精确替换文件中的一段文本
bash(cmd) — 在工作目录执行 shell 命令（bash）

工作规则：
- 每轮输出恰好一个工具调用，格式：TOOL: 名称(参数json)。例如：
  TOOL: read({"path": "src/app.py"})
  TOOL: bash({"cmd": "python -c 'print(1)'"})
- 我会返回工具结果，你继续下一步。
- 完成任务后输出：DONE（单独一行）。
- 最多 {max_steps} 轮。任务描述见首轮 user 消息。"""


def llm_call(base_url, api_key, model, messages, max_tokens=4096, timeout=300):
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 1.0}).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    msg = d["choices"][0]["message"]
    return msg.get("content") or (msg.get("reasoning_content") or "")


import re
TOOL_RE = re.compile(r'TOOL:\s*(\w+)\s*\((\{.*\})\)\s*$', re.S)

def parse_tool(text):
    m = TOOL_RE.search(text.strip())
    if not m: return None
    try:
        return m.group(1), json.loads(m.group(2))
    except json.JSONDecodeError:
        return None


def run_tool(name, args, cwd):
    try:
        if name == "read":
            p = (cwd / args["path"]).resolve()
            if not str(p).startswith(str(cwd.resolve())): return "ERROR: path escapes workdir"
            return p.read_text(errors="replace")[:8000]
        if name == "write":
            p = (cwd / args["path"]).resolve()
            if not str(p).startswith(str(cwd.resolve())): return "ERROR: path escapes workdir"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"])
            return f"wrote {args['path']} ({len(args['content'])} chars)"
        if name == "edit":
            p = (cwd / args["path"]).resolve()
            if not str(p).startswith(str(cwd.resolve())): return "ERROR: path escapes workdir"
            t = p.read_text(errors="replace")
            if args["old"] not in t: return "ERROR: old text not found"
            if t.count(args["old"]) > 1: return "ERROR: old text not unique"
            p.write_text(t.replace(args["old"], args["new"], 1))
            return "edited"
        if name == "bash":
            r = subprocess.run(["bash", "-c", args["cmd"]], cwd=cwd,
                               capture_output=True, text=True, timeout=120)
            out = (r.stdout or "") + (r.stderr or "")
            return f"exit={r.returncode}\n{out[:6000]}"
    except Exception as e:
        return f"ERROR: {e}"
    return "ERROR: unknown tool"


def run_agent(base_url, api_key, model, task_dir, max_steps=30):
    """跑一个任务的一次 rollout，返回 trajectory。"""
    cwd = Path(tempfile.mkdtemp(prefix="station-task-"))
    traj = []
    try:
        if (task_dir / "setup").is_dir():
            for f in (task_dir / "setup").iterdir():
                if f.is_file(): shutil.copy(f, cwd / f.name)
                elif f.is_dir(): shutil.copytree(f, cwd / f.name)
        instruction = (task_dir / "instruction.md").read_text()
        messages = [{"role": "system", "content": SYSTEM_PROMPT.replace("{max_steps}", str(max_steps))},
                    {"role": "user", "content": instruction + "\n\n（工作目录中已放置任务初始文件，先查看再动手。完成后输出 DONE）"}]
        for step in range(max_steps):
            # 网络/服务端错误：指数退避重试 3 次
            reply = None
            for attempt in range(3):
                try:
                    reply = llm_call(base_url, api_key, model, messages)
                    break
                except Exception as e:
                    traj.append({"step": step, "event": "llm_error",
                                 "attempt": attempt, "error": str(e)})
                    time.sleep(2 ** attempt)
            if reply is None:
                # 连续 API 失败：记 RUN_ERROR，非能力失败
                traj.append({"step": step, "event": "run_error", "error": "api_unavailable"})
                return traj, "RUN_ERROR"
            # 空回复（可能只有 reasoning 被截断）：注入提示重试一轮，避免空 assistant 污染历史
            if not reply.strip():
                messages.append({"role": "assistant", "content": "(继续，请直接输出工具调用)"})
                messages.append({"role": "user", "content": "你的上一条回复为空。请输出 TOOL: 名称({json参数}) 或 DONE。"})
                traj.append({"step": step, "event": "empty_reply"})
                continue
            messages.append({"role": "assistant", "content": reply})
            traj.append({"step": step, "assistant": reply})
            if reply.strip().startswith("DONE") or reply.strip().endswith("DONE"):
                break
            tool = parse_tool(reply)
            if not tool:
                obs = "ERROR: 输出格式应为 TOOL: 名称({json参数}) 或 DONE。请重新输出。"
            else:
                obs = run_tool(tool[0], tool[1], cwd)
            messages.append({"role": "user", "content": str(obs)})
            traj.append({"step": step, "observation": str(obs)[:2000]})
        # verify
        v = subprocess.run(["bash", str(task_dir / "verify.sh")], cwd=cwd,
                           capture_output=True, text=True, timeout=180)
        passed = v.returncode == 0
        traj.append({"event": "verify", "passed": passed, "stdout": v.stdout[-1500:],
                     "stderr": v.stderr[-1500:]})
        return traj, passed
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


def validate_tasks(tasks_dir):
    """Verifier 防抖：参考解跑 3 遍，必须全 pass；空环境跑 1 遍，必须 fail。"""
    ok = True
    for tdir in sorted(Path(tasks_dir).iterdir()):
        if not (tdir / "verify.sh").exists():
            print(f"[SKIP] {tdir.name}: no verify.sh"); continue
        # 参考解 3 遍
        results = []
        for i in range(3):
            with tempfile.TemporaryDirectory() as td:
                cwd = Path(td)
                if (tdir / "setup").is_dir():
                    shutil.copytree(tdir / "setup", cwd, dirs_exist_ok=True)
                shutil.copytree(tdir / "solution", cwd, dirs_exist_ok=True)
                v = subprocess.run(["bash", str(tdir / "verify.sh")], cwd=cwd,
                                   capture_output=True, text=True, timeout=120)
                results.append(v.returncode == 0)
        # 空环境 1 遍（应 fail，防 no-op 通过）
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            if (tdir / "setup").is_dir():
                shutil.copytree(tdir / "setup", cwd, dirs_exist_ok=True)
            v = subprocess.run(["bash", str(tdir / "verify.sh")], cwd=cwd,
                               capture_output=True, text=True, timeout=120)
            noop_fail = v.returncode != 0
        status = "OK " if (all(results) and noop_fail) else "BAD"
        if status == "BAD": ok = False
        print(f"[{status}] {tdir.name}: solution x3 = {results}, no-op fail = {noop_fail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(HARNESS_DIR.parent / "tasks"))
    ap.add_argument("--models", default=os.environ.get("EVAL_MODELS", "deepseek-flash"))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--base-url", default=os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:3001"))
    ap.add_argument("--api-key", default=os.environ.get("EVAL_API_KEY"))
    ap.add_argument("--out", default=str(HARNESS_DIR.parent / "results"))
    ap.add_argument("--list-tasks", action="store_true")
    ap.add_argument("--validate-tasks", action="store_true")
    args = ap.parse_args()

    tasks = sorted(d for d in Path(args.tasks).resolve().iterdir() if (d / "verify.sh").exists())

    if args.list_tasks:
        for t in tasks: print(t.name)
        return
    if args.validate_tasks:
        sys.exit(0 if validate_tasks(args.tasks) else 1)

    api_key = args.api_key
    if not api_key:  # 从 new-api 数据库读 hermes-linux token
        import sqlite3
        db = sqlite3.connect("/opt/new-api/one-api.db")
        api_key = db.execute("select key from tokens where name='hermes-linux'").fetchone()[0]

    outdir = Path(args.out); outdir.mkdir(exist_ok=True)
    all_results = {}
    for model in args.models.split(","):
        model_results = {}
        for tdir in tasks:
            passes = []
            run_errors = 0
            for r in range(args.repeats):
                t0 = time.time()
                traj, passed = run_agent(args.base_url, api_key, model.strip(), tdir, args.max_steps)
                if passed == "RUN_ERROR":
                    run_errors += 1
                    passed = False
                    print(f"{model.strip()} / {tdir.name} / r{r}: RUN_ERROR ({time.time()-t0:.0f}s)", flush=True)
                else:
                    print(f"{model.strip()} / {tdir.name} / r{r}: {'PASS' if passed else 'FAIL'} ({time.time()-t0:.0f}s)", flush=True)
                passes.append(passed)
                (outdir / f"{model.strip()}__{tdir.name}__r{r}.json").write_text(
                    json.dumps(traj, ensure_ascii=False, indent=1))
            model_results[tdir.name] = passes
            if run_errors:
                print(f"  ⚠ {tdir.name}: {run_errors}/{args.repeats} RUN_ERROR（API 故障，不计能力失败）", flush=True)
        all_results[model.strip()] = model_results

    # 汇总
    print("\n" + "=" * 60)
    summary = {}
    for model, res in all_results.items():
        flat = [p for v in res.values() for p in v]
        n, c = len(flat), sum(flat)
        p = c / n if n else 0
        lo, hi = wilson_ci(p, n)
        pk = pass_at_k(flat, min(args.repeats, n))
        summary[model] = {"trials": n, "passes": c, "pass_rate": round(p, 3),
                          "ci95": [round(lo, 3), round(hi, 3)],
                          f"pass@{args.repeats}": round(pk, 3),
                          f"pass^{args.repeats}": round(
                              sum(1 for k, v in res.items() if all(v)) / len(res), 3)}
        print(f"{model}: {c}/{n} = {p:.1%}  CI95 [{lo:.2f},{hi:.2f}]  "
              f"pass@{args.repeats}={pk:.2f}  pass^{args.repeats}={summary[model][f'pass^{args.repeats}']}")
    (outdir / "summary.json").write_text(json.dumps(
        {"models": summary, "repeats": args.repeats,
         "tasks": {m: {t: v for t, v in r.items()} for m, r in all_results.items()}},
        ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
