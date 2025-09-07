import pandas as pd
from src.data import fetch as fetch_mod

HTML = """
<table>
  <thead>
    <tr>
      <th>Date</th><th>Visitor/Neutral</th><th>PTS</th>
      <th>Home/Neutral</th><th>PTS.1</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>2024-10-20</td><td>BOS</td><td>101</td><td>NYK</td><td>99</td></tr>
    <tr><td>2024-10-21</td><td>LAL</td><td>110</td><td>GSW</td><td>108</td></tr>
  </tbody>
</table>
"""

def test_end_to_end_writes_csv(tmp_path, monkeypatch):
    # point OUT_DIR to tmp
    monkeypatch.setattr(fetch_mod, "OUT_DIR", tmp_path, raising=True)

    # make fetch_season_html return canned HTML
    monkeypatch.setattr(fetch_mod, "fetch_season_html", lambda yr: HTML, raising=True)

    # run for two seasons (content identical, de-duped by game_id)
    fetch_mod.main([2024, 2025])

    out_csv = tmp_path / "games.csv"
    assert out_csv.exists()

    df = pd.read_csv(out_csv, parse_dates=["GAME_DATE"])
    assert len(df) == 2
    assert df["game_id"].nunique() == 2
    assert {"GAME_DATE","home_team","away_team","home_score","away_score","home_win"}.issubset(df.columns)
