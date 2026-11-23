"""Smoke test: every dashboard page renders in demo mode (no loan-level data), as on a public deployment.

Runs in a subprocess so the dashboard's module-level config (pointed at empty data directories) and Streamlit's
caches don't leak into other tests.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("dashboard").resolve()))
from streamlit.testing.v1 import AppTest

results = {}
for page in ["dashboard/app.py", *sorted(str(p) for p in Path("dashboard/pages").glob("*.py"))]:
    at = AppTest.from_file(str(Path(page).resolve()), default_timeout=180).run()
    errors = [str(e.value) for e in at.exception]
    if "Strategy" in page:
        at.slider[-1].set_value(0.10).run()                 # collections capacity
        errors += [str(e.value) for e in at.exception]
        results["capture_labels"] = [m.label for m in at.metric]
    if "Monitoring" in page:
        at.radio[0].set_value("Simulated drift").run()
        errors += [str(e.value) for e in at.exception]
        results["monitoring_labels"] = [m.label for m in at.metric]
    results[Path(page).name] = errors

# Website mode (?site=1): no sidebar, page controls move into the page.
site = AppTest.from_file(str(Path("dashboard/pages/1_Portfolio_Overview.py").resolve()), default_timeout=180)
site.query_params["site"] = "1"
site.run()
results["site_mode"] = {"errors": [str(e.value) for e in site.exception],
                        "sidebar_sliders": len(site.sidebar.slider), "main_sliders": len(site.main.slider),
                        "sidebar_text": len(site.sidebar.caption) + len(site.sidebar.markdown)}
print("RESULTS" + json.dumps(results))
"""


@pytest.mark.skipif(not (ROOT / "artifacts" / "model" / "MLmodel").exists(), reason="no exported champion")
def test_dashboard_pages_render_in_demo_mode(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text())
    cfg["data"].update(raw_path=str(tmp_path / "missing.csv"), processed_dir=str(tmp_path / "processed"),
                       reference_dir=str(tmp_path / "reference"))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    env = {**os.environ, "CREDIT_RISK_CONFIG": str(tmp_path / "config.yaml"), "MLFLOW_DISABLE_AGENT_HINT": "1"}
    env.pop("API_URL", None)

    out = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    results = json.loads(out.stdout.split("RESULTS", 1)[1])

    pages = {k: v for k, v in results.items() if k.endswith(".py")}
    assert len(pages) == 6
    assert all(not errors for errors in pages.values()), pages
    assert "Prioritisation lift" in results["capture_labels"]
    # Simulated borrowers have no outcomes, so no observed default rate may be shown for them.
    assert "Observed default rate" not in results["monitoring_labels"]
    # Embedded in the website: the sidebar is empty and the LGD control sits in the page.
    assert results["site_mode"] == {"errors": [], "sidebar_sliders": 0, "main_sliders": 1, "sidebar_text": 0}


@pytest.mark.skipif(not (ROOT / "artifacts" / "model" / "MLmodel").exists(), reason="no exported champion")
def test_simulated_portfolio_is_realistic_but_imperfect():
    """The demo portfolio follows the real PD deciles, and the model works well on it, but less well than on the
    real test set (AUC 0.769), as on any new population."""
    from sklearn.metrics import roc_auc_score

    from credit_risk import load_config
    from credit_risk.predict import load_model, score_frame, simulated_portfolio

    cfg = load_config()
    model = load_model(str(ROOT / "artifacts" / "model"), cfg)
    df = simulated_portfolio(model, 2000, seed=3, cfg=cfg)
    p = score_frame(model, df, cfg)["probability_of_default"].to_numpy()
    y = df[cfg["data"]["target"]].to_numpy()
    assert len(df) == 2000 and not df["SK_ID_CURR"].duplicated().any()
    assert set(y) <= {0, 1}
    assert abs(p.mean() - 0.08) < 0.01                         # real test set: mean PD 8.0%
    assert 0.0 < y.mean() - p.mean() < 0.04                    # mild under-prediction (about 8.0% vs 9.4%)
    assert 0.64 < roc_auc_score(y, p) < 0.76                   # good but below the real 0.769
