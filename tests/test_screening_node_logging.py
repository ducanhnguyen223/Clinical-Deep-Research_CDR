"""Regression tests for operational logging in screening pipeline nodes."""

import hashlib
import logging

import pytest

from cdr.core.enums import RecordSource
from cdr.core.schemas import PICO, CDRState, Record, ScreeningDecision
from cdr.orchestration.nodes.screening_nodes import parse_documents_node, screen_node


@pytest.fixture
def pico():
    return PICO(
        population="adults with condition",
        intervention="treatment X",
        outcome="symptom relief",
    )


def make_record(record_id: str, abstract: str) -> Record:
    title = f"Study {record_id}"
    return Record(
        record_id=record_id,
        title=title,
        abstract=abstract,
        source=RecordSource.PUBMED,
        content_hash=hashlib.sha256(f"{title}|{abstract}".encode()).hexdigest(),
    )


@pytest.mark.asyncio
async def test_screen_node_logs_blocked_research_screening_as_error(pico, caplog, capsys):
    state = CDRState(
        run_id="screen-log-test",
        question="Does treatment X improve symptom relief?",
        pico=pico,
    )

    caplog.set_level(logging.ERROR, logger="cdr.orchestration.nodes.screening_nodes")
    result = await screen_node(state, {"configurable": {"dod_level": 2}})

    assert any(
        record.levelno == logging.ERROR
        and "DoD Level 2" in record.getMessage()
        and "heuristic screening is insufficient" in record.getMessage()
        for record in caplog.records
    )
    assert any("LLM screening required for DoD Level 2" in error for error in result["errors"])
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_screen_node_logs_exploratory_heuristic_fallback_as_warning(pico, caplog, capsys):
    state = CDRState(
        run_id="screen-fallback-log-test",
        question="Does treatment X improve symptom relief?",
        pico=pico,
    )

    caplog.set_level(logging.DEBUG, logger="cdr.orchestration.nodes.screening_nodes")
    result = await screen_node(
        state,
        {"configurable": {"dod_level": 1, "enable_evidence_gates": False}},
    )

    assert any(
        record.levelno == logging.WARNING
        and "heuristic screening" in record.getMessage()
        and "Level 1" in record.getMessage()
        for record in caplog.records
    )
    assert any(
        record.levelno == logging.DEBUG and "Included: 0, Excluded: 0" in record.getMessage()
        for record in caplog.records
    )
    assert result["screened"] == []
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_parse_documents_logs_missing_content_with_record_id_and_length(caplog, capsys):
    record = make_record("rec-short", "Too short")
    state = CDRState(
        run_id="parse-log-test",
        question="Does treatment X improve symptom relief?",
        retrieved_records=[record],
        screened=[ScreeningDecision(record_id=record.record_id, included=True)],
    )

    caplog.set_level(logging.WARNING, logger="cdr.orchestration.nodes.screening_nodes")
    result = await parse_documents_node(
        state,
        {"configurable": {"enable_fulltext_retrieval": False}},
    )

    assert any(
        record.levelno == logging.WARNING
        and "rec-short" in record.getMessage()
        and "text length: 9" in record.getMessage()
        and "reports_not_retrieved" in record.getMessage()
        for record in caplog.records
    )
    assert result["prisma_counts"].reports_not_retrieved == 1
    assert result["snippets"] == []
    assert capsys.readouterr().out == ""
