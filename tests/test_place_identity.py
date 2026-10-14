"""The identity check: a multi-brain run corroborates its scans rather than contradicting them."""

from __future__ import annotations

import pytest

from datapipe.place import Identity, _id_tokens, check_identity


def test_id_tokens_keeps_only_the_parts_that_name_an_animal():
    assert _id_tokens("HL_GPe_161_202") == {"161", "202"}
    assert _id_tokens("HL_t013_t017_FPOF10") == {"t013", "t017", "fpof10"}
    # words shared by every scan of a rig say nothing about which animal it is
    assert _id_tokens("HL_GPe_nameless") == set()


@pytest.mark.parametrize(
    ("recipe", "run"),
    [
        ("t013", "HL_t013_t017_FPOF10"),  # two brains, split per animal afterwards
        ("sample_161", "HL_GPe_161_202"),  # the cage number is the shared identifier
        ("T4", "HL_GPe_T4_T6"),
        ("T11", "HL_T11_T12_T14"),  # three brains in one run
        ("HL_GPe160", "HL_LM_L_R_GPe160_GPe174"),
        ("HL_t008_205", "HL_t008_205"),  # one brain: the names are simply equal
    ],
)
def test_a_run_carrying_the_recipe_id_is_consistent(recipe, run):
    ident = Identity(recipe_id=recipe, acqlog_ids={run})
    assert ident.unreconciled == set()
    assert ident.internally_consistent


@pytest.mark.parametrize(
    ("recipe", "run"),
    [
        ("t013", "HL_s121_A323"),  # a different animal entirely
        ("t013", "HL_seq130"),  # 130 must not reconcile with t013 through a bare 13
        ("EJT38", "EJT48"),  # the mislabelling this check exists for
        (
            "HL_GPe_nameless",
            "HL_GPe_other",
        ),  # no identifier in either: nothing to reconcile on
    ],
)
def test_a_run_naming_another_animal_stays_a_disagreement(recipe, run):
    ident = Identity(recipe_id=recipe, acqlog_ids={run})
    assert ident.unreconciled == {run}
    assert not ident.internally_consistent


def test_without_a_recipe_two_run_names_are_still_ambiguous():
    assert Identity(acqlog_ids={"HL_t013"}).internally_consistent
    ident = Identity(acqlog_ids={"HL_t013", "HL_t017"})
    assert not ident.internally_consistent
    assert ident.unreconciled == {"HL_t013", "HL_t017"}


def test_check_identity_still_refuses_the_wrong_subject_and_says_why():
    ident = Identity(recipe_id="t013", acqlog_ids={"HL_t013_t017_FPOF10"})
    problems = check_identity(ident, "t013_acute_m1102591_211", roster=None)
    assert len(problems) == 1
    assert "different animals" in problems[0]
    assert "disagrees with itself" not in problems[0]


def test_check_identity_accepts_the_subject_a_roster_relates_to_the_scan(tmp_path):
    roster = tmp_path / "roster.yaml"
    roster.write_text(
        "subjects:\n"
        "- lgcy_id: t013_acute_m1102591_211\n"
        "  src_names: [t013, HL_t013_t017_FPOF10]\n"
    )
    ident = Identity(recipe_id="t013", acqlog_ids={"HL_t013_t017_FPOF10"})
    assert check_identity(ident, "t013_acute_m1102591_211", roster=roster) == []
    # and the roster does not widen it to a subject it does not name
    assert check_identity(ident, "t017_acute_m1102265_213", roster=roster)


def test_a_scan_that_states_no_identity_is_refused():
    problems = check_identity(Identity(), "t013_acute_m1102591_211", roster=None)
    assert len(problems) == 1
    assert "states no sample identity" in problems[0]
