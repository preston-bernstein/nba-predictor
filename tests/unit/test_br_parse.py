import pytest

from src.data.br_parse import parse_games

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
    <!-- header repeat row -->
    <tr><td>Date</td><td>Visitor/Neutral</td><td>PTS</td><td>Home/Neutral</td><td>PTS.1</td></tr>
    <tr><td>2024-10-21</td><td>LAL</td><td>110</td><td>GSW</td><td>108</td></tr>
    <!-- postponed -->
    <tr><td>2024-10-22</td><td>MIA</td><td></td><td>CHI</td><td></td></tr>
  </tbody>
</table>
"""


def test_parse_games_basic():
    df = parse_games(HTML)
    # columns present
    expect = {
        "GAME_DATE",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_win",
        "game_id",
    }
    assert expect.issubset(df.columns)

    # rows: postponed dropped; duplicate deduped -> 2 unique games
    assert len(df) == 2

    r0 = df.iloc[0]
    assert str(r0["GAME_DATE"].date()) == "2024-10-20"
    assert r0["home_team"] == "NYK"
    assert r0["away_team"] == "BOS"
    assert r0["home_score"] == 99
    assert r0["away_score"] == 101
    assert r0["home_win"] == 0


def test_parse_games_invalid_raises():
    with pytest.raises(ValueError):
        parse_games("<html><body>No tables here</body></html>")


def test_parse_games_raises_when_table_missing_required_columns():
    # Has a <table>, but headers don't include REQUIRED_COLS → triggers your line-13 branch
    HTML_BAD = """
    <table>
      <thead>
        <tr>
          <th>NotDate</th><th>Visitor?</th><th>Points</th>
          <th>Home?</th><th>Points2</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>2024-10-20</td><td>BOS</td><td>101</td><td>NYK</td><td>99</td></tr>
      </tbody>
    </table>
    """
    from pytest import raises

    from src.data.br_parse import parse_games

    with raises(ValueError, match="No game tables found"):
        parse_games(HTML_BAD)
