import pytest

from gmq.universe import InstrumentLink, Role, Segment, UniverseScope, context_feature_namespace, linked_context


def test_equity_is_primary_and_other_segments_are_context():
    scope = UniverseScope()
    assert scope.is_primary(Segment.EQUITY)
    assert not scope.admissible_for_trade(Segment.EQUITY_DERIVATIVES)
    assert scope.is_contextual(Segment.CURRENCY)
    assert scope.required_segments() == {
        Segment.EQUITY,
        Segment.EQUITY_DERIVATIVES,
        Segment.CURRENCY,
        Segment.COMMODITY,
    }


def test_context_links_are_ranked_and_not_trade_targets():
    links = [
        InstrumentLink("RELIANCE", "RELIANCE", Segment.EQUITY, Role.PRIMARY, 1.0),
        InstrumentLink("RELIANCE", "RELIANCE-FUT", Segment.EQUITY_DERIVATIVES, relevance=0.9),
        InstrumentLink("RELIANCE", "USDINR", Segment.CURRENCY, relevance=0.4),
        InstrumentLink("TCS", "USDINR", Segment.CURRENCY, relevance=0.8),
    ]
    ctx = linked_context("reliance", links)
    assert [x.instrument_symbol for x in ctx] == ["RELIANCE-FUT", "USDINR"]
    assert all(x.role is Role.CONTEXT for x in ctx)
    assert context_feature_namespace(ctx[0]) == "equity_derivatives.RELIANCE-FUT"


def test_relevance_range_is_validated():
    with pytest.raises(ValueError):
        InstrumentLink("TCS", "USDINR", Segment.CURRENCY, relevance=1.1)
