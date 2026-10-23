"""Self-contained HTML report."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment

from .data import describe_value

_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Labour informality in Paraguay</title>
<style>
 :root{--bg:#fff;--fg:#16191d;--muted:#5c6672;--line:#e3e7ec;--zebra:#f7f9fb;
  --pass:#17795e;--pass-bg:#e6f4ef;--warn:#8a5a00;--warn-bg:#fdf3e0;
  --fail:#b02a1f;--fail-bg:#fdecea;--accent:#1f4e79;--bar:#4a86c5;--bar2:#c5563f}
 @media(prefers-color-scheme:dark){:root{--bg:#14171a;--fg:#e8ecf1;--muted:#9aa5b1;
  --line:#2a2f36;--zebra:#191d21;--pass:#4cc79f;--pass-bg:#16302a;--warn:#e0a33c;
  --warn-bg:#33280f;--fail:#f1786c;--fail-bg:#3a1d1a;--accent:#7fb2e5;--bar:#5d9bd8;
  --bar2:#d4705a}}
 *{box-sizing:border-box}
 body{margin:0;padding:2rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
 .wrap{max-width:1060px;margin:0 auto}
 h1{font-size:1.6rem;margin:0 0 .25rem;letter-spacing:-.01em}
 h2{font-size:1.05rem;margin:2.4rem 0 .7rem}
 .sub{color:var(--muted);margin:0 0 1.8rem;font-size:.9rem}
 .cards{display:grid;gap:.75rem;grid-template-columns:repeat(auto-fit,minmax(155px,1fr))}
 .card{border:1px solid var(--line);border-radius:8px;padding:.9rem 1rem}
 .card .n{font-size:1.5rem;font-weight:650;line-height:1.1}
 .card .l{color:var(--muted);font-size:.75rem;text-transform:uppercase;
  letter-spacing:.04em;margin-top:.2rem}
 .card.ok .n{color:var(--pass)} .card.bad .n{color:var(--fail)}
 .scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
 table{border-collapse:collapse;width:100%;font-size:.86rem}
 th,td{text-align:right;padding:.5rem .7rem;border-bottom:1px solid var(--line);white-space:nowrap}
 th:first-child,td:first-child{text-align:left;white-space:normal}
 th{font-weight:600;color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.04em}
 tbody tr:nth-child(even){background:var(--zebra)}
 tbody tr:last-child td{border-bottom:none}
 code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.92em}
 .pill{display:inline-block;padding:.08rem .45rem;border-radius:4px;font-size:.72rem;font-weight:650}
 .pill.pass{color:var(--pass);background:var(--pass-bg)}
 .pill.error{color:var(--fail);background:var(--fail-bg)}
 .pill.warning{color:var(--warn);background:var(--warn-bg)}
 .pill.info{color:var(--muted);background:var(--line)}
 .note{border-left:3px solid var(--accent);padding:.55rem 0 .55rem .9rem;color:var(--muted);
  margin:1rem 0;font-size:.89rem}
 .bar{height:11px;background:var(--bar);border-radius:2px;display:inline-block;vertical-align:middle}
 .bar.alt{background:var(--bar2)}
 footer{margin-top:3rem;color:var(--muted);font-size:.79rem;border-top:1px solid var(--line);padding-top:1rem}
</style></head><body><div class="wrap">
 <h1>Labour informality in Paraguay</h1>
 <p class="sub">{{ universe_label }} &middot; EPHC {{ year }} microdata, Instituto Nacional
  de Estad&iacute;stica &middot; generated {{ generated_at }}</p>

 <div class="cards">
  <div class="card"><div class="n">{{ rate }}</div><div class="l">Informality rate</div></div>
  <div class="card"><div class="n">{{ unweighted }}</div><div class="l">If weights ignored</div></div>
  <div class="card"><div class="n">{{ deff }}&times;</div><div class="l">Interval widening</div></div>
  <div class="card"><div class="n">{{ n_eff }}</div><div class="l">Effective sample</div></div>
  <div class="card"><div class="n">{{ best_auc }}</div><div class="l">Model AUC</div></div>
  <div class="card {{ 'ok' if summary.blocking == 0 else 'bad' }}">
   <div class="n">{{ summary.passed }}/{{ summary.total }}</div><div class="l">Checks passed</div></div>
 </div>

 <div class="note">
  The survey is a clustered probability sample. Every rate here is weighted by
  the expansion factor, and every interval comes from resampling whole primary
  sampling units rather than rows &mdash; which is why they are wider than a
  textbook binomial interval would be.
 </div>

 <h2>The headline estimate</h2>
 <div class="scroll"><table>
  <thead><tr><th>Measure</th><th>Value</th><th>Meaning</th></tr></thead><tbody>
  <tr><td>Weighted rate</td><td><strong>{{ rate }}</strong></td>
   <td style="text-align:left">{{ population }} workers</td></tr>
  <tr><td>Unweighted rate</td><td>{{ unweighted }}</td>
   <td style="text-align:left">what an analysis ignoring the design would report</td></tr>
  <tr><td>Naive 95% interval</td><td>{{ naive_ci }}</td>
   <td style="text-align:left">binomial, assumes independent observations</td></tr>
  <tr><td>Design-based 95% interval</td><td><strong>{{ design_ci }}</strong></td>
   <td style="text-align:left">cluster bootstrap, {{ replicates }} replicates</td></tr>
  <tr><td>Respondents / clusters</td><td>{{ n }} / {{ clusters }}</td>
   <td style="text-align:left">effective sample size {{ n_eff }}</td></tr>
  </tbody></table></div>

 <h2>How the sample narrows</h2>
 <div class="scroll"><table>
  <thead><tr><th>Step</th><th>Remaining</th></tr></thead><tbody>
  {% for label, n in universe_steps %}<tr><td>{{ label }}</td><td>{{ '{:,}'.format(n) }}</td></tr>{% endfor %}
  </tbody></table></div>

 {% for block in breakdowns %}
 <h2>{{ block.label }}</h2>
 <div class="scroll"><table>
  <thead><tr><th>Category</th><th>n</th><th>Informality</th><th>95% interval</th><th></th></tr></thead><tbody>
  {% for row in block.rows %}
  <tr><td>{{ row.label }}</td><td>{{ '{:,}'.format(row.n) }}</td>
   {% if row.suppressed %}<td colspan="3" style="text-align:left;color:var(--muted)">
     too few respondents to estimate</td>
   {% else %}
   <td>{{ '%.1f'|format(row.rate*100) }}%</td>
   <td>{{ '%.1f'|format(row.low*100) }}&ndash;{{ '%.1f'|format(row.high*100) }}%</td>
   <td style="width:30%"><span class="bar" style="width:{{ row.rate*100 }}%"></span></td>
   {% endif %}</tr>
  {% endfor %}
  </tbody></table></div>
 {% endfor %}

 <h2>Prediction</h2>
 <p class="sub">Five-fold cross-validation, weighted. The clustered split keeps every
  primary sampling unit entirely on one side, so the model is scored on places it
  has never seen.</p>
 <div class="scroll"><table>
  <thead><tr><th>Model</th><th>AUC (clustered)</th><th>AUC (random split)</th>
   <th>Optimism</th><th>Brier</th><th>Baseline Brier</th></tr></thead><tbody>
  {% for m in models %}
  <tr><td>{{ m.name }}</td>
   <td><strong>{{ '%.4f'|format(m.clustered_auc) }}</strong></td>
   <td>{{ '%.4f'|format(m.random_auc) }}</td>
   <td>{{ '%+.4f'|format(m.optimism) }}</td>
   <td>{{ '%.4f'|format(m.brier) }}</td>
   <td>{{ '%.4f'|format(m.baseline_brier) }}</td></tr>
  {% endfor %}
  </tbody></table></div>

 <h2>What the model is actually using</h2>
 <p class="sub">Each feature removed, the model refitted, and the AUC it gave up measured.</p>
 <div class="scroll"><table>
  <thead><tr><th>Feature</th><th>AUC without it</th><th>Lost</th><th></th></tr></thead><tbody>
  {% for c in contributions %}
  <tr><td>{{ c.label }} <code>{{ c.feature }}</code></td>
   <td>{{ '%.4f'|format(c.auc_without) }}</td>
   <td>{{ '%+.4f'|format(c.loss) }}</td>
   <td style="width:40%">{% if c.loss > 0 %}<span class="bar alt"
     style="width:{{ (c.loss / max_loss * 100) if max_loss else 0 }}%"></span>{% endif %}</td></tr>
  {% endfor %}
  </tbody></table></div>

 <h2>Checks</h2>
 <div class="scroll"><table>
  <thead><tr><th>Status</th><th>Check</th><th>Expected</th><th>Observed</th></tr></thead><tbody>
  {% for c in checks %}
  <tr><td>{% if c.passed %}<span class="pill pass">pass</span>
      {% else %}<span class="pill {{ c.severity }}">{{ c.severity }}</span>{% endif %}</td>
   <td><code>{{ c.check_id }}</code>{% if c.detail %}
     <div style="color:var(--muted)">{{ c.detail }}</div>{% endif %}</td>
   <td style="text-align:left">{{ c.expected }}</td>
   <td style="text-align:left">{{ c.observed }}</td></tr>
  {% endfor %}
  </tbody></table></div>

 <footer>Built from INE microdata. Every figure is computed by the run that produced
  this page. Not an official statistic of the Instituto Nacional de Estad&iacute;stica.</footer>
</div></body></html>
"""


def build_context(analysis) -> dict:
    cfg = analysis.config
    headline = analysis.headline

    breakdowns = []
    for column, rows in analysis.by_group.items():
        breakdowns.append({
            "label": analysis.labels.get(column, column),
            "rows": [
                {**row, "label": describe_value(analysis.value_labels, column, row["group"])}
                for row in rows
            ],
        })

    models = [{
        "name": name.replace("_", " "),
        "clustered_auc": m.mean("clustered", "auc"),
        "random_auc": m.mean("random", "auc"),
        "optimism": m.optimism,
        "brier": m.mean("clustered", "brier"),
        "baseline_brier": m.baseline_brier,
    } for name, m in analysis.models.items()]

    contributions = [{
        "feature": c.feature,
        "label": analysis.labels.get(c.feature, c.feature),
        "auc_without": c.auc_without,
        "loss": c.loss,
    } for c in analysis.contributions]

    return {
        "year": cfg["source"]["primary_year"],
        "universe_label": cfg["universe"]["employment_category"]["label"],
        "rate": f"{headline.value:.1%}",
        "unweighted": f"{headline.unweighted:.1%}",
        "deff": f"{headline.design_effect:.2f}",
        "n": f"{headline.n:,}",
        "n_eff": f"{headline.effective_sample_size:,}",
        "clusters": f"{headline.clusters:,}",
        "population": f"{headline.population:,.0f}",
        "naive_ci": f"{headline.naive_low:.1%}–{headline.naive_high:.1%}",
        "design_ci": f"{headline.design_low:.1%}–{headline.design_high:.1%}",
        "replicates": cfg["bootstrap"]["replicates"],
        "universe_steps": analysis.universe.steps,
        "breakdowns": breakdowns,
        "models": models,
        "best_auc": f"{analysis.best_model.mean('clustered', 'auc'):.3f}",
        "contributions": contributions,
        "max_loss": max((c["loss"] for c in contributions), default=0.0),
        "checks": analysis.checks,
        "summary": analysis.summary,
    }


def render(analysis, output_path: Path) -> Path:
    html = Environment(autoescape=True).from_string(_TEMPLATE).render(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        **build_context(analysis),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
