"""Run tearsheet — self-contained HTML one-pager for a single run.

UAT §5 #14: 'Export to PDF / PPT tearsheet. A one-click investor
tearsheet with charts, metrics, recipe hash, audit status. Saves the
analyst 2 hours every Friday.'

V1 ships HTML only. PDF generation requires weasyprint or a headless
browser, both of which add deployment complexity (system-level
dependencies for weasyprint, Chromium binaries for headless). Browser
'Save as PDF' from the rendered HTML produces a clean static copy
that's good enough for a Friday memo. Phase-3 ships server-side PDF
when there's actual demand.

The HTML is self-contained: no external CSS, no external fonts, no
external images. Charts are embedded as base64-encoded PNGs pulled
from the executed notebook's display_data outputs (same algorithm as
the run-page <RunCharts/> component).

Layout mirrors the run-detail page in the same reading order:
header → headline traffic-light → audit/hypothesis verdicts → charts
→ metric grid → walk-forward stability → regime breakdown →
reproducibility → recipe YAML.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Iterable

import nbformat


# ─── Detection / extraction ──────────────────────────────────────────


def _read_notebook(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return nbformat.read(f, as_version=4)
    except Exception:
        return None


def _extract_chart_pngs(nb: dict) -> list[tuple[str | None, str]]:
    """Return (caption, base64-png) tuples for each chart in the notebook.

    Same heuristic as src/components/run-charts.tsx — match cells whose
    finagent.node_id is in {n10_charts, n11_decomposition} OR fall back
    to any code cell with image/png outputs. Caption pulled from the
    preceding markdown step header when available.
    """
    out: list[tuple[str | None, str]] = []
    if not nb:
        return out
    cells = nb.get("cells") or []
    prev_md: str | None = None
    for cell in cells:
        ctype = cell.get("cell_type")
        if ctype == "markdown":
            src = cell.get("source") or ""
            prev_md = "".join(src) if isinstance(src, list) else src
            continue
        if ctype != "code":
            continue
        # PNG-bearing cells include the chart steps and any future template
        # that emits image/png outputs.
        pngs: list[str] = []
        for outp in cell.get("outputs") or []:
            if outp.get("output_type") not in ("display_data", "execute_result"):
                continue
            png = (outp.get("data") or {}).get("image/png")
            if isinstance(png, str) and png:
                pngs.append(png.strip())
            elif isinstance(png, list) and png:
                pngs.append("".join(png).strip())
        if not pngs:
            prev_md = None
            continue
        # Caption: rationale meta if present, else markdown step title.
        cap: str | None = None
        meta = cell.get("metadata") or {}
        fmeta = meta.get("finagent")
        if isinstance(fmeta, dict) and fmeta.get("rationale"):
            cap = str(fmeta["rationale"])
        elif prev_md:
            for line in prev_md.splitlines():
                line = line.strip()
                if line.startswith("###"):
                    cap = line.lstrip("#").strip()
                    break
        for p in pngs:
            out.append((cap, p))
        prev_md = None
    return out


# ─── HTML helpers ────────────────────────────────────────────────────


def _e(s: Any) -> str:
    return html.escape(str(s)) if s is not None else "—"


def _fmt_ratio(v: Any) -> str:
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        return "—"
    if abs(v) >= 1e3:
        return f"implausible ({v:.2e})"
    return f"{v:.4f}"


def _fmt_return_pct(v: Any) -> str:
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        return "—"
    if abs(v) >= 1e3:
        return f"implausible ({v:.2e})"
    sign = "+" if v > 0 else ""
    return f"{sign}{v * 100:.2f}%"


def _fmt_drawdown(v: Any) -> str:
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        return "—"
    return f"{v * 100:+.2f}%"


def _fmt_pct_raw(v: Any) -> str:
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        return "—"
    return f"{v * 100:.1f}%"


# Routes a metric key (possibly with model_/value_/etc. prefix and
# _net/_gross suffix) to the right formatter.
_BOOK_PREFIXES = ("model_", "value_", "momentum_", "buy_and_hold_", "both_legs_")
_SUFFIXES = ("_net", "_gross")


def _strip(key: str) -> str:
    s = key
    for p in _BOOK_PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
            break
    for sfx in _SUFFIXES:
        if s.endswith(sfx):
            s = s[: -len(sfx)]
            break
    return s


def _fmt_metric(key: str, v: Any) -> str:
    base = _strip(key)
    if base in ("annual_return", "total_return"):
        return _fmt_return_pct(v)
    if base == "max_drawdown":
        return _fmt_drawdown(v)
    if base == "hit_rate":
        return _fmt_pct_raw(v)
    return _fmt_ratio(v)


# ─── Section builders ────────────────────────────────────────────────


_BASE_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: #f8fafc;
  color: #0f172a;
  margin: 0;
  padding: 32px;
  line-height: 1.5;
}
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; font-weight: 600; }
h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
     color: #64748b; margin: 32px 0 12px; font-weight: 600; }
.muted { color: #64748b; }
.mono { font-family: "SF Mono", Menlo, Consolas, monospace; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-family: "SF Mono", Menlo, monospace; font-size: 10.5px;
        text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; margin-right: 6px; }
.pill.green  { background: #d1fae5; color: #065f46; }
.pill.amber  { background: #fef3c7; color: #92400e; }
.pill.rose   { background: #fee2e2; color: #991b1b; }
.pill.grey   { background: #e2e8f0; color: #475569; }
.headline { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;
            padding: 20px; border-radius: 10px; margin-top: 16px; }
.headline.green { background: #ecfdf5; border: 1px solid #a7f3d0; }
.headline.amber { background: #fffbeb; border: 1px solid #fde68a; }
.headline.rose  { background: #fef2f2; border: 1px solid #fecaca; }
.headline.grey  { background: #f1f5f9; border: 1px solid #cbd5e1; }
.headline .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
                   color: #64748b; font-weight: 600; }
.headline .value { font-size: 28px; font-weight: 700; font-family: "SF Mono", Menlo, monospace;
                   margin-top: 4px; }
.value.implausible { color: #991b1b; }
.book { padding: 14px 16px; border: 1px solid #e2e8f0; border-radius: 8px;
        background: #fff; margin-bottom: 12px; }
.book h3 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
           color: #475569; margin: 0 0 10px; font-weight: 600; }
.book.benchmark { opacity: 0.85; background: #f8fafc; }
.book .grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px 16px; }
@media (max-width: 800px) { .book .grid { grid-template-columns: repeat(2, 1fr); } }
.kv { display: flex; justify-content: space-between; align-items: baseline;
      font-size: 12px; gap: 12px; }
.kv .k { color: #64748b; font-family: "SF Mono", monospace; }
.kv .v { color: #0f172a; font-family: "SF Mono", monospace; }
.kv .v.implausible { color: #991b1b; }
.kv .gross { font-size: 10.5px; color: #94a3b8; text-align: right; }
table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }
thead { background: #f1f5f9; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #e2e8f0; }
th { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; }
td.num { text-align: right; font-family: "SF Mono", monospace; }
img.chart { max-width: 100%; height: auto; display: block; margin: 0 auto; }
figure { margin: 0 0 18px; padding: 14px; border: 1px solid #e2e8f0; border-radius: 8px;
         background: #fff; }
figcaption { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
             color: #64748b; margin-bottom: 10px; font-weight: 600; }
.audit-block { padding: 14px 16px; border-radius: 8px; margin-top: 8px;
               border: 1px solid; background: #fff; }
.audit-block.PASS    { border-color: #a7f3d0; background: #ecfdf5; }
.audit-block.FLAGGED { border-color: #fde68a; background: #fffbeb; }
.audit-block.FAILED  { border-color: #fecaca; background: #fef2f2; }
.audit-block.FAIL    { border-color: #fecaca; background: #fef2f2; }
.audit-block.CANCEL  { border-color: #fde68a; background: #fffbeb; }
.audit-block ul { margin: 8px 0 0; padding-left: 0; list-style: none; }
.audit-block li { font-size: 12px; margin: 4px 0; }
.severity { display: inline-block; padding: 1px 6px; border-radius: 3px;
            font-size: 9.5px; font-family: "SF Mono", monospace;
            text-transform: uppercase; margin-right: 6px; vertical-align: middle; }
.severity.critical { background: #fee2e2; color: #991b1b; }
.severity.warning  { background: #fef3c7; color: #92400e; }
.severity.info     { background: #e2e8f0; color: #475569; }
.severity.pass     { background: #d1fae5; color: #065f46; }
.severity.fail     { background: #fee2e2; color: #991b1b; }
pre.yaml { background: #0f172a; color: #f1f5f9; padding: 16px; border-radius: 8px;
           font-size: 11.5px; line-height: 1.6; white-space: pre-wrap;
           word-break: break-word; overflow-x: auto; }
.repro { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px 24px;
         padding: 14px 16px; border: 1px solid #e2e8f0; border-radius: 8px;
         background: #fff; }
@media (max-width: 700px) { .repro { grid-template-columns: 1fr; } }
.repro .label { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.08em;
                color: #94a3b8; font-weight: 600; }
.repro .val { font-size: 12px; margin-top: 4px; word-break: break-word; }
.lib-pill { display: inline-block; background: #f1f5f9; color: #475569; padding: 1px 6px;
            border-radius: 3px; font-family: "SF Mono", monospace; font-size: 10.5px;
            margin: 2px 4px 2px 0; }
.note { font-size: 11px; color: #94a3b8; margin-top: 4px; font-style: italic; }
@media print {
  body { background: white; padding: 16px; }
  .no-print { display: none; }
}
"""


def _traffic_for(run: dict) -> tuple[str, str]:
    """Same traffic-light rule as the run-detail page. Returns (tone, label)."""
    if run.get("status") == "failed":
        return ("rose", "Run failed")
    if run.get("status") != "completed":
        return ("grey", run.get("status", "queued"))
    audit = run.get("bias_audit") or {}
    if audit.get("verdict") == "FAILED":
        return ("rose", "Audit failed")
    flag_count = len(run.get("metrics_flags") or {})
    if audit.get("verdict") == "FLAGGED" or flag_count > 0:
        return ("amber", "Metrics flagged" if flag_count else "Audit flagged")
    if not audit:
        return ("grey", "Audit pending")
    return ("green", "Looks good")


def _section_headline(run: dict) -> str:
    metrics = run.get("metrics") or {}
    flags = run.get("metrics_flags") or {}
    tone, label = _traffic_for(run)
    flag_count = len(flags)
    costs_applied = any(k.endswith("_net") for k in metrics)

    def _stat(metric_key: str, lbl: str, fmt) -> str:
        v = metrics.get(metric_key)
        flag = flags.get(metric_key)
        text = fmt(v)
        cls = "value implausible" if "implausible" in text else "value"
        title = f' title="{_e(flag)}"' if flag else ""
        return (
            f'<div{title}><div class="label">{_e(lbl)}</div>'
            f'<div class="{cls}">{_e(text)}{" ⚠" if flag else ""}</div></div>'
        )

    chips = ""
    if costs_applied:
        chips += '<span class="pill grey">Net of costs</span>'
    if flag_count:
        chips += f'<span class="pill amber">{flag_count} metric{"" if flag_count==1 else "s"} flagged</span>'

    return f"""
<div class="headline {tone}">
  <div style="grid-column: 1 / -1; display:flex; justify-content:space-between; align-items:center;">
    <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.06em; color:#475569; font-weight:600;">{_e(label)}</div>
    <div>{chips}</div>
  </div>
  {_stat('annual_return', 'Annual return', _fmt_return_pct)}
  {_stat('sharpe', 'Sharpe', _fmt_ratio)}
  {_stat('max_drawdown', 'Max drawdown', _fmt_drawdown)}
</div>
"""


def _section_audit(run: dict) -> str:
    audit = run.get("bias_audit")
    if not audit:
        return ""
    verdict = audit.get("verdict", "PENDING")
    summary = audit.get("summary", "")
    reasons = audit.get("reasons") or []
    items = "".join(
        f'<li><span class="severity {_e(r.get("severity","info"))}">{_e(r.get("severity",""))}</span>'
        f'<strong>{_e(r.get("check_name",""))}:</strong> {_e(r.get("evidence",""))}</li>'
        for r in reasons
    )
    return f"""
<h2>Bias audit · {_e(verdict)}</h2>
<div class="audit-block {_e(verdict)}">
  <p style="margin:0;">{_e(summary)}</p>
  {f'<ul>{items}</ul>' if items else ''}
</div>
"""


def _section_hypothesis(run: dict) -> str:
    h = run.get("hypothesis_verdict")
    if not h:
        return ""
    verdict = h.get("verdict", "PENDING")
    thesis = h.get("thesis", "")
    summary = h.get("summary", "")
    checks = h.get("checks") or []
    rows = ""
    for c in checks:
        crit = c.get("criterion") or {}
        actual = c.get("actual")
        kind = c.get("kind", "success")
        passed = bool(c.get("passed"))
        if kind == "cancel":
            chip_cls = "fail" if passed else "info"
            chip_label = "cancel hit" if passed else "cancel ok"
        else:
            chip_cls = "pass" if passed else "fail"
            chip_label = "pass" if passed else "fail"
        actual_s = "—" if actual is None else f"{actual:.4f}"
        rows += (
            f'<li><span class="severity {chip_cls}">{_e(chip_label)}</span>'
            f'<strong>{_e(crit.get("metric",""))}</strong> '
            f'{_e(crit.get("op",""))} <strong>{_e(crit.get("value",""))}</strong>'
            f' → actual {_e(actual_s)}</li>'
        )
    return f"""
<h2>Hypothesis · {_e(verdict)}</h2>
<div class="audit-block {_e(verdict)}">
  <p style="margin:0;">{_e(summary)}</p>
  <p style="margin:8px 0 0; font-style:italic; color:#475569;">&ldquo;{_e(thesis)}&rdquo;</p>
  {f'<ul>{rows}</ul>' if rows else ''}
</div>
"""


def _section_charts(notebook_path: Path) -> str:
    nb = _read_notebook(notebook_path)
    charts = _extract_chart_pngs(nb) if nb else []
    if not charts:
        return ""
    body = "".join(
        f'<figure>{f"<figcaption>{_e(cap)}</figcaption>" if cap else ""}'
        f'<img class="chart" src="data:image/png;base64,{p}" alt="{_e(cap or "chart")}"/></figure>'
        for cap, p in charts
    )
    return f"<h2>Charts</h2>{body}"


def _section_metric_grid(run: dict) -> str:
    metrics = run.get("metrics") or {}
    flags = run.get("metrics_flags") or {}
    HEADLINE = {"annual_return", "sharpe", "max_drawdown"}
    KNOWN = ["model", "value", "momentum", "both_legs", "buy_and_hold"]

    grouped: dict[str, list[str]] = {}
    for k in metrics:
        if k in HEADLINE:
            continue
        bucket = "other"
        for b in KNOWN:
            if k.startswith(b + "_"):
                bucket = b
                break
        grouped.setdefault(bucket, []).append(k)

    ORDER = ("annual_return", "total_return", "sharpe", "sortino",
             "calmar", "max_drawdown", "turnover", "hit_rate", "exposure")

    def _pair_book(book: str, title: str, tone: str) -> str:
        keys = grouped.get(book, [])
        if not keys:
            return ""
        # Pair gross + net.
        pairs: dict[str, dict[str, str]] = {}
        for k in keys:
            base = _strip(k)
            slot = pairs.setdefault(base, {})
            if k.endswith("_net"):
                slot["net"] = k
            else:
                slot["gross"] = k
        ordered = sorted(pairs.keys(), key=lambda b: ORDER.index(b) if b in ORDER else 999)
        rows = ""
        for base in ordered:
            slot = pairs[base]
            primary = slot.get("net") or slot.get("gross")
            if not primary:
                continue
            v_text = _fmt_metric(primary, metrics.get(primary))
            v_cls = "v implausible" if "implausible" in v_text else "v"
            flag = flags.get(primary)
            v_attr = f' title="{_e(flag)}"' if flag else ""
            gross_caption = ""
            if "net" in slot and "gross" in slot:
                g = _fmt_metric(slot["gross"], metrics.get(slot["gross"]))
                gross_caption = f'<div class="gross">gross {_e(g)}</div>'
            rows += (
                f'<div><div class="kv"><span class="k">{_e(base)}</span>'
                f'<span class="{v_cls}"{v_attr}>{_e(v_text)}{" ⚠" if flag else ""}</span></div>'
                f'{gross_caption}</div>'
            )
        return f'<div class="book {tone}"><h3>{_e(title)}</h3><div class="grid">{rows}</div></div>'

    blocks = []
    blocks.append(_pair_book("model", "Strategy", ""))
    components = ""
    for b, t in [("value", "Value"), ("momentum", "Momentum"), ("both_legs", "Both legs")]:
        components += _pair_book(b, t, "")
    if components:
        blocks.append(f'<div>{components}</div>')
    bench = _pair_book("buy_and_hold", "Buy & hold (benchmark — not strategy performance)", "benchmark")
    if bench:
        blocks.append(bench)
    other = _pair_book("other", "Other", "")
    if other:
        blocks.append(other)
    body = "".join(b for b in blocks if b)
    if not body:
        return ""
    return f"<h2>Metrics</h2>{body}"


def _section_fold_stability(run: dict) -> str:
    folds = run.get("fold_metrics") or []
    if not folds:
        return ""
    rows = ""
    for f in folds:
        rows += (
            f'<tr><td>{_e(f.get("fold"))}</td>'
            f'<td>{_e(f.get("start") or "—")} → {_e(f.get("end") or "—")}</td>'
            f'<td class="num">{_e(f.get("n_obs") or "—")}</td>'
            f'<td class="num">{_e(_fmt_ratio(f.get("sharpe")))}</td>'
            f'<td class="num">{_e(_fmt_return_pct(f.get("annual_return")))}</td>'
            f'<td class="num">{_e(_fmt_drawdown(f.get("max_drawdown")))}</td></tr>'
        )
    return f"""
<h2>Walk-forward stability</h2>
<table><thead><tr><th>Fold</th><th>Window</th><th>n</th><th>Sharpe</th>
<th>Annual</th><th>Max DD</th></tr></thead><tbody>{rows}</tbody></table>
"""


def _section_regime_breakdown(run: dict) -> str:
    regimes = run.get("regime_metrics") or []
    if not regimes:
        return ""
    rows = ""
    for r in regimes:
        rows += (
            f'<tr><td><span class="pill grey">state {_e(r.get("regime"))}</span></td>'
            f'<td>{_e(r.get("n_obs"))} obs · {_e(_fmt_pct_raw(r.get("pct_of_oos")))}</td>'
            f'<td class="num">{_e(_fmt_ratio(r.get("sharpe")))}</td>'
            f'<td class="num">{_e(_fmt_return_pct(r.get("annual_return")))}</td>'
            f'<td class="num">{_e(_fmt_drawdown(r.get("max_drawdown")))}</td></tr>'
        )
    return f"""
<h2>Regime-conditional performance</h2>
<table><thead><tr><th>Regime</th><th>Coverage</th><th>Sharpe</th>
<th>Annual</th><th>Max DD</th></tr></thead><tbody>{rows}</tbody></table>
"""


def _section_repro(run: dict, notebook_path: Path) -> str:
    nb = _read_notebook(notebook_path)
    rmeta = (nb.get("metadata") or {}).get("finagent_recipe") or {} if nb else {}
    seed = rmeta.get("seed")
    libs = rmeta.get("library_versions") or {}
    vintage = rmeta.get("data_vintage") or {}
    libs_html = "".join(
        f'<span class="lib-pill">{_e(k)} {_e(v)}</span>'
        for k, v in libs.items() if v
    ) or "—"

    def _vintage_line(var: str, d: dict) -> str:
        bits: list[str] = []
        if d.get("kind"):
            bits.append(f" ({_e(d.get('kind'))})")
        if d.get("start"):
            bits.append(f" · from {_e(d.get('start'))}")
        if d.get("end"):
            bits.append(f" · to {_e(d.get('end'))}")
        return f'<li class="mono" style="margin:2px 0;"><strong>{_e(var)}</strong>{"".join(bits)}</li>'

    vint_html = "".join(_vintage_line(var, d) for var, d in vintage.items()) or "<li>—</li>"
    return f"""
<h2>Reproducibility</h2>
<div class="repro">
  <div><div class="label">Recipe hash</div><div class="val mono">{_e(run.get("recipe_hash"))}</div></div>
  <div><div class="label">Seed</div><div class="val mono">{_e(seed if seed is not None else "—")}</div></div>
  <div><div class="label">Compiled at</div><div class="val mono">{_e(rmeta.get("compiled_at") or "—")}</div></div>
  <div style="grid-column:1/-1;"><div class="label">Library versions</div><div class="val">{libs_html}</div></div>
  <div style="grid-column:1/-1;"><div class="label">Data vintage</div><ul style="margin:4px 0 0; padding-left:18px;">{vint_html}</ul></div>
</div>
"""


def _section_recipe(run: dict) -> str:
    return f"<h2>Recipe</h2><pre class=\"yaml\">{_e(run.get('recipe_yaml') or '')}</pre>"


def render_tearsheet(run: dict, notebook_path: Path | None) -> str:
    """Build a self-contained HTML tearsheet for a run.

    `run` must be the public dict shape returned by Run.as_public_dict
    (includes metrics, metrics_flags, bias_audit, hypothesis_verdict,
    fold_metrics, regime_metrics, recipe_yaml, etc.). `notebook_path`
    is the on-disk path to the executed notebook, used to extract chart
    PNGs and reproducibility metadata.
    """
    title = run.get("name") or run.get("id", "run")
    subtitle_parts = []
    if run.get("template"):
        subtitle_parts.append(run["template"])
    if run.get("recipe_hash"):
        subtitle_parts.append(f"#{run['recipe_hash'][:10]}")
    subtitle = " · ".join(subtitle_parts)

    sections: list[str] = [
        f'<h1>{_e(title)}</h1>',
        f'<div class="muted mono" style="font-size:12px;">{_e(subtitle)}</div>',
        _section_headline(run),
        _section_hypothesis(run),
        _section_audit(run),
    ]
    if notebook_path is not None and notebook_path.exists():
        sections.append(_section_charts(notebook_path))
    sections.extend([
        _section_metric_grid(run),
        _section_fold_stability(run),
        _section_regime_breakdown(run),
    ])
    if notebook_path is not None and notebook_path.exists():
        sections.append(_section_repro(run, notebook_path))
    sections.append(_section_recipe(run))
    sections.append(
        '<p class="note no-print">Use your browser&rsquo;s Save as PDF '
        '(File → Print → Save as PDF) to capture a static copy.</p>'
    )

    body = "\n".join(s for s in sections if s)
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Tearsheet — {_e(title)}</title>
<style>{_BASE_CSS}</style>
</head>
<body><div class="wrap">
{body}
</div></body></html>"""
