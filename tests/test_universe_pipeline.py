from gmq.universe.pipeline import SecuritySnapshot, UniverseFilterConfig, UniversePipeline


def test_pipeline_keeps_all_usable_knowledge_but_filters_tradability():
    rows = [
        SecuritySnapshot("TCS", last_price=3500, avg_daily_value=100_000_000, spread_bps=5,
                         data_quality=1.0, volatility=0.12, signal_score=.8),
        SecuritySnapshot("ILLIQUID", last_price=100, avg_daily_value=1_000_000, spread_bps=120,
                         data_quality=1.0, volatility=.10, signal_score=.9),
        SecuritySnapshot("HALTED", last_price=1000, avg_daily_value=100_000_000, spread_bps=4,
                         data_quality=1.0, volatility=.10, signal_score=.9, halted=True),
    ]
    result = UniversePipeline().run(rows)
    assert result.counts == {"knowledge": 3, "tradable": 1, "candidates": 1}
    assert result.tradable[0].symbol == "TCS"


def test_candidates_are_ranked_and_capped():
    cfg = UniverseFilterConfig(min_avg_daily_value=1, max_spread_bps=100,
                               min_data_quality=.5, max_volatility=1.0,
                               min_candidate_score=0.0, max_candidates=2)
    rows = [
        SecuritySnapshot("A", avg_daily_value=10, spread_bps=1, signal_score=.3),
        SecuritySnapshot("B", avg_daily_value=10, spread_bps=1, signal_score=.9),
        SecuritySnapshot("C", avg_daily_value=10, spread_bps=1, signal_score=.6),
    ]
    result = UniversePipeline(cfg).run(rows)
    assert [x.symbol for x in result.candidates] == ["B", "C"]


def test_non_eligible_security_stays_out_of_knowledge():
    rows = [SecuritySnapshot("OK"), SecuritySnapshot("BAD", instrument_eligible=False)]
    result = UniversePipeline().run(rows)
    assert [x.symbol for x in result.knowledge] == ["OK"]
