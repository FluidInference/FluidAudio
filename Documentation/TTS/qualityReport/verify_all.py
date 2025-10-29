#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import coremltools as ct  # type: ignore
except Exception:
    ct = None  # type: ignore


def _rss_kb() -> int:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])  # type: ignore[name-defined]
        return int(out.decode().strip())
    except Exception:
        return 0


def sh(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    env = os.environ.copy()
    # Ensure KOKORO_PY_SRC is set for scripts that need the Kokoro package
    if "KOKORO_PY_SRC" not in env:
        # Look for kokoro in common locations
        repo_root = Path(__file__).resolve().parents[3]
        candidates = [
            repo_root / "kokoro",
            repo_root.parent / "mobius" / "kokoro",
        ]
        for candidate in candidates:
            if (candidate / "kokoro").exists():
                env["KOKORO_PY_SRC"] = str(candidate)
                break
    # Set HF cache to use mobius models directory to avoid downloads
    if "HF_HOME" not in env:
        mobius_cache = repo_root.parent / "mobius" / "model_cache"
        if mobius_cache.exists():
            env["HF_HOME"] = str(mobius_cache)
    # Suppress the "No module named pip" message that uv Python prints on startup
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    # Filter out the pip warning from stderr
    stderr_lines = [line for line in result.stderr.split('\n') if 'No module named pip' not in line]
    filtered_stderr = '\n'.join(stderr_lines)
    if filtered_stderr.strip():
        print(filtered_stderr, file=sys.stderr, end='')
    if result.stdout:
        print(result.stdout, end='')
    if result.returncode != 0:
        if not filtered_stderr.strip():
            # No error shown, print original stderr for debugging
            print(f"[debug] Original stderr:\n{result.stderr}", file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)


def main() -> None:
    # Everything is in the qualityReport directory
    qual_dir = Path(__file__).resolve().parent
    scripts_dir = qual_dir.parent / "scripts"
    out_dir = qual_dir / "perf_out" / "batch_technical_obstacles"
    out_dir.mkdir(parents=True, exist_ok=True)

    TEXT = (
        "The technical obstacles you'll face have to do with the need to develop a "
        "technology that is at the same time compatible with, and yet superior to, existing products."
    )
    VOICE = "af_heart"

    wav_ref = out_dir / "pytorch.wav"
    wav_onnx = out_dir / "onnx.wav"
    wav_v24_15s = out_dir / "v24_15s.wav"
    wav_v21_10s = out_dir / "v21_10s.wav"
    wav_v24_10s = out_dir / "v24_10s.wav"

    onnx_path = qual_dir / "kokoro-v1.0.onnx"
    v24_15s_pkg = qual_dir / "kokoro_24_15s.mlpackage"
    v21_10s_pkg = qual_dir / "kokoro_21_10s.mlpackage"
    v24_10s_pkg = qual_dir / "kokoro_24_10s.mlpackage"

    print("== PyTorch (reference) ==")
    try:
        sh([
            sys.executable,
            str(scripts_dir / "run-pytorch-infer.py"),
            "--text",
            TEXT,
            "--voice",
            VOICE,
            "--out",
            str(wav_ref),
        ])
    except subprocess.CalledProcessError:
        print("[warn] PyTorch inference failed (skipping reference). This is optional if you only want to compare CoreML models.")

    if onnx_path.exists():
        print("== ONNX ==")
        try:
            sh([
                sys.executable,
                str(scripts_dir / "run-onnx-infer.py"),
                "--text",
                TEXT,
                "--voice",
                VOICE,
                "--out",
                str(wav_onnx),
            ])
        except subprocess.CalledProcessError:
            print("[warn] ONNX step failed (likely onnxruntime not installed). Skipping…")
    else:
        print(f"[skip] ONNX not found at {onnx_path}")

    if v24_15s_pkg.exists():
        print("== CoreML v24 15s ==")
        try:
            sh([
                sys.executable,
                str(scripts_dir / "run-coreml-infer.py"),
                "--mlpackage",
                str(v24_15s_pkg),
                "--text",
                TEXT,
                "--voice",
                VOICE,
                "--out",
                str(wav_v24_15s),
            ])
        except subprocess.CalledProcessError:
            print("[warn] CoreML v24 15s failed (skipping).")
    if v21_10s_pkg.exists():
        print("== CoreML v21 10s ==")
        try:
            sh([
                sys.executable,
                str(scripts_dir / "run-coreml-infer.py"),
                "--mlpackage",
                str(v21_10s_pkg),
                "--text",
                TEXT,
                "--voice",
                VOICE,
                "--out",
                str(wav_v21_10s),
            ])
        except subprocess.CalledProcessError:
            print("[warn] CoreML v21 10s failed (skipping).")
    if v24_10s_pkg.exists():
        print("== CoreML v24 10s ==")
        try:
            sh([
                sys.executable,
                str(scripts_dir / "run-coreml-infer.py"),
                "--mlpackage",
                str(v24_10s_pkg),
                "--text",
                TEXT,
                "--voice",
                VOICE,
                "--out",
                str(wav_v24_10s),
            ])
        except subprocess.CalledProcessError:
            print("[warn] CoreML v24 10s failed (skipping).")

    candidates = [p for p in [wav_onnx, wav_v24_10s, wav_v24_15s, wav_v21_10s] if p.exists()]
    if not candidates:
        print("[warn] No candidates generated; stopping before metrics.")
        return

    # Only run metrics if we have a reference
    if wav_ref.exists():
        print("== Metrics (core set + CLAP + sim_index) ==")
        metrics_csv = out_dir / "metrics_core_set_plus.csv"
        sh([
            sys.executable,
            str(scripts_dir / "benchmark_core_set.py"),
            "--ref",
            str(wav_ref),
            "--cands",
            *[str(c) for c in candidates],
            "--csv",
            str(metrics_csv),
        ])
    else:
        print("[skip] No PyTorch reference wav; skipping metrics comparison.")

    # Print quick ranking by sim_index
    try:
        rows = []
        with metrics_csv.open() as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    rows.append((Path(row["candidate"]).name, float(row.get("sim_index", "nan"))))
                except Exception:
                    pass
        rows = [x for x in rows if x[1] == x[1]]
        rows.sort(key=lambda x: x[1], reverse=True)
        print("-- Similarity Index ranking (desc) --")
        for n, s in rows:
            print(f"{n}: {s:.4f}")
    except Exception as e:
        print(f"[warn] could not summarize sim_index: {e}")

    # Write a Markdown table with key metrics
    try:
        with metrics_csv.open() as f:
            r = csv.DictReader(f)
            metrics_rows = list(r)
        md_lines = []
        md_lines.append("| Model | ECAPA | CLAP | MFCC-cos | MFCC-cos (norm) | LSD | LSD (norm) | MCD dB | F0 RMSE | F0 r | Dur (s) | Drift % | sim_index |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in metrics_rows:
            name = Path(row.get("candidate", "")).name
            def f(k: str) -> float:
                try:
                    return float(row.get(k, "nan"))
                except Exception:
                    return float('nan')
            ecapa = f("ecapa_cos"); clap = f("clap_cos")
            mfcc = f("mfcc_cos_dtw"); mfccn = f("mfcc_cos_dtw_norm")
            lsd = f("lsd_dtw"); lsdn = f("lsd_dtw_norm")
            mcd = f("mcd_db"); f0rmse = f("f0_rmse_cents"); f0r = f("f0_r_dtw")
            dur = f("dur_cand_s"); drift = f("dur_pct") * 100.0 if f("dur_pct") == f("dur_pct") else float('nan')
            sim = f("sim_index")
            md_lines.append(
                f"| {name} | {ecapa:.3f} | {clap:.3f} | {mfcc:.3f} | {mfccn:.3f} | {lsd:.3f} | {lsdn:.3f} | {mcd:.3f} | {f0rmse:.3f} | {f0r:.3f} | {dur:.3f} | {drift:.3f} | {sim:.3f} |"
            )
        md_path = out_dir / "metrics_table.md"
        md_text = "\n".join(md_lines) + "\n"
        md_path.write_text(md_text, encoding="utf-8")
        print(f"[ok] Wrote {md_path}")

        # Build a single-file report that includes metrics table and (if present) speed/memory
        report_lines = []
        report_lines.append("# Kokoro Verify Report")
        report_lines.append("")
        report_lines.append("## Similarity Index (desc)")
        if rows:
            for n, s in rows:
                report_lines.append(f"- {n}: {s:.4f}")
        else:
            report_lines.append("- (no candidates)")
        report_lines.append("")
        report_lines.append("## Core Metrics")
        report_lines.append(md_text)
        speed_csv = out_dir / "speed_mem_summary.csv"
        if speed_csv.exists():
            try:
                report_lines.append("")
                report_lines.append("## Speed / Memory")
                with speed_csv.open() as sf:
                    rr = list(csv.DictReader(sf))
                rep = []
                rep.append("| Model | avg_s | audio_s | RTFx | RSS (MB) |")
                rep.append("|---|---:|---:|---:|---:|")
                for r0 in rr:
                    rep.append(
                        f"| {r0['model']} | {r0['avg_s']} | {r0['audio_s']} | {r0['rtfx']} | {r0['rss_mb']} |"
                    )
                report_lines.extend(rep)
            except Exception as e:
                report_lines.append(f"(speed/memory unavailable: {e})")
        report_path = out_dir / "report.md"
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        print(f"[ok] Wrote {report_path}")
    except Exception as e:
        print(f"[warn] could not write metrics_table.md: {e}")

    # 3) Speed/RAM (RTFx + peak RSS) — default output
    try:
        import os
        import torch  # type: ignore

        def _resolve_model_path(p: Path) -> Path:
            p = p.resolve()
            if p.is_dir() and p.suffix == ".mlpackage":
                inner = p / "Data/com.apple.CoreML/model.mlmodel"
                if inner.exists():
                    return inner
            return p

        def _required_tokens(mlp: Path) -> int:
            if ct is None:
                return 242
            mlmodel = ct.models.MLModel(str(_resolve_model_path(mlp)))
            spec = mlmodel.get_spec()
            for i in spec.description.input:
                if i.name == "input_ids":
                    t = i.type.multiArrayType
                    if len(t.shape) == 2:
                        return int(t.shape[1])
            return 242

        def _make_style(voice_pt: Path, pick: int = 128) -> np.ndarray:
            pack = torch.load(str(voice_pt), map_location="cpu", weights_only=True)
            pick = max(0, min(pack.shape[0] - 1, pick))
            return pack[pick].squeeze(0).numpy().astype(np.float32)

        def _make_inputs(tokens: int, ref_s: np.ndarray) -> dict[str, np.ndarray]:
            ids = np.ones((1, tokens), dtype=np.int32)
            mask = np.ones((1, tokens), dtype=np.int32)
            rph = np.zeros((1, 9), dtype=np.float32)
            return {
                "input_ids": ids,
                "ref_s": ref_s[None, :].astype(np.float32) if ref_s.ndim == 1 else ref_s.astype(np.float32),
                "random_phases": rph,
                "attention_mask": mask,
            }

        def _bench_coreml(mlp: Path, inputs: dict[str, np.ndarray], runs: int = 6) -> tuple[float, float, int]:
            if ct is None:
                raise RuntimeError("coremltools not available")
            cu = getattr(ct.ComputeUnit, "CPU_AND_NE", ct.ComputeUnit.ALL)
            mlmodel = ct.models.MLModel(str(_resolve_model_path(mlp)), compute_units=cu)
            # warmup
            for _ in range(2):
                mlmodel.predict(inputs)
            max_rss = 0
            t0 = time.perf_counter()
            out = None
            for _ in range(runs):
                out = mlmodel.predict(inputs)
                try:
                    rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).decode().strip())  # type: ignore[name-defined]
                    max_rss = max(max_rss, rss)
                except Exception:
                    pass
            t1 = time.perf_counter()
            avg = (t1 - t0) / runs
            alen = None
            if out is not None:
                for _, v in out.items():
                    arr = np.array(v)
                    if arr.dtype.kind in "iu":
                        alen = int(arr.ravel()[0])
                        break
            audio_s = (alen / 24000.0) if alen is not None else 0.0
            return avg, audio_s, max_rss

        def _bench_onnx(onnx_p: Path, ref_s: np.ndarray, runs: int = 6) -> tuple[float, float, int]:
            try:
                import onnxruntime as ort  # type: ignore
                if not hasattr(ort, "InferenceSession"):
                    from onnxruntime.capi import InferenceSession as _IS  # type: ignore
                    class _Shim:
                        InferenceSession = _IS
                    ort = _Shim()  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"onnxruntime not available: {e}")
            ids = np.ones((1, 242), dtype=np.int64)
            feed = {"input_ids": ids, "style": ref_s[None, :].astype(np.float32), "speed": np.array([1], dtype=np.int32)}
            sess = ort.InferenceSession(str(onnx_p), providers=["CPUExecutionProvider"])
            for _ in range(2):
                sess.run(None, feed)
            max_rss = 0
            t0 = time.perf_counter()
            for _ in range(runs):
                out = sess.run(None, feed)
                rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).decode().strip())  # type: ignore[name-defined]
                max_rss = max(max_rss, rss)
            t1 = time.perf_counter()
            wav = np.array(out[0]).ravel()
            audio_s = wav.size / 24000.0
            avg = (t1 - t0) / runs
            return avg, audio_s, max_rss

        voice_pt = qual_dir / "voices" / "af_heart.pt"
        if voice_pt.exists():
            ref = _make_style(voice_pt)
            rows = []
            # CoreML v24 10s
            if v24_10s_pkg.exists():
                tokens = _required_tokens(v24_10s_pkg)
                avg, audio_s, rss = _bench_coreml(v24_10s_pkg, _make_inputs(tokens, ref))
                rows.append(("coreml_v24_10s", avg, audio_s, (audio_s / avg) if avg else 0.0, rss / 1024.0))
            # CoreML v24 15s
            if v24_15s_pkg.exists():
                tokens = _required_tokens(v24_15s_pkg)
                avg, audio_s, rss = _bench_coreml(v24_15s_pkg, _make_inputs(tokens, ref))
                rows.append(("coreml_v24_15s", avg, audio_s, (audio_s / avg) if avg else 0.0, rss / 1024.0))
            # CoreML v21 10s
            if v21_10s_pkg.exists():
                tokens = _required_tokens(v21_10s_pkg)
                avg, audio_s, rss = _bench_coreml(v21_10s_pkg, _make_inputs(tokens, ref))
                rows.append(("coreml_v21_10s", avg, audio_s, (audio_s / avg) if avg else 0.0, rss / 1024.0))
            # ONNX (optional)
            try:
                if onnx_path.exists():
                    avg, audio_s, rss = _bench_onnx(onnx_path, ref)
                    rows.append(("onnx", avg, audio_s, (audio_s / avg) if avg else 0.0, rss / 1024.0))
            except Exception as e:
                print(f"[warn] Skipping ONNX RTFx/memory: {e}")

            # Write CSV
            speed_csv = out_dir / "speed_mem_summary.csv"
            with speed_csv.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["model", "avg_s", "audio_s", "rtfx", "rss_mb"])  # rss_mb = peak resident set size
                for r in rows:
                    w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.3f}", f"{r[3]:.2f}", f"{r[4]:.1f}"])
            print(f"[ok] Wrote {speed_csv}")
            for r in rows:
                print(r[0], {"avg_s": round(r[1], 4), "audio_s": round(r[2], 3), "rtfx": round(r[3], 2), "rss_mb": round(r[4], 1)})
        else:
            print(f"[skip] RTFx/memory: voice pack not found at {voice_pt}")
    except Exception as e:
        print(f"[warn] RTFx/memory step failed: {e}")

    print(f"[ok] Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
