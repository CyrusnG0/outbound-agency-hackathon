import pytest
from pydantic import ValidationError

from app.schemas import CompanyProfile, Signal, ICPAssessment, PolicyGateDecision, ReplyClassification


def test_company_profile_requires_summary_and_confidence():
    profile = CompanyProfile(
        one_line_summary="B2B software for logistics teams",
        industry="Logistics SaaS",
        estimated_size="51-200",
        geo="US",
        confidence=0.82,
    )
    assert profile.one_line_summary == "B2B software for logistics teams"
    with pytest.raises(ValidationError):
        CompanyProfile(one_line_summary="x", confidence=1.5)  # out of [0,1]


def test_signal_type_is_constrained_to_scoring_rules_values():
    sig = Signal(
        signal_type="hiring_relevant_role",
        signal_value="Hiring operations managers",
        signal_strength=0.73,
        source_url="https://example.com/careers",
        source_confidence=0.87,
        # B2a: evidence_quote is a required field on every Signal, so this
        # valid-construction fixture must provide one.
        evidence_quote="We are hiring an operations manager for our team",
    )
    assert sig.signal_strength == 0.73
    assert sig.evidence_quote == "We are hiring an operations manager for our team"
    with pytest.raises(ValidationError):
        # The quote is present so the ONLY reason this raises is the invalid
        # signal_type — the test stays about the taxonomy, not the new field.
        Signal(
            signal_type="not_a_real_type", signal_value="x", signal_strength=0.5,
            evidence_quote="We are hiring an operations manager for our team",
        )


def test_icp_assessment_fit_label_constrained():
    assessment = ICPAssessment(
        fit_label="good_fit",
        fit_score=78,
        fit_reasons=["Operations-heavy workflow"],
        non_fit_reasons=[],
    )
    assert assessment.fit_score == 78
    with pytest.raises(ValidationError):
        ICPAssessment(fit_label="maybe_fit", fit_score=78, fit_reasons=[], non_fit_reasons=[])


def test_reply_classification_is_constrained():
    """C1: the classifier's output is constrained at the schema layer —
    an invented class fails validation (the Literal refuses anything
    outside the nine reply-routing.md §1 classes), an out-of-range
    confidence fails, and short rationale/quote fail the minimum-length
    floors (the judge_icp.py discipline, re-applied)."""
    verdict = ReplyClassification(
        reply_class="positive",
        confidence=0.8,
        rationale=(
            "The reply asks for more detail about the offer, which is the "
            "positive class's signature — no other class fits its wording."
        ),
        evidence_quote="Could you send over a bit more detail",
    )
    assert verdict.reply_class == "positive"
    with pytest.raises(ValidationError):
        # An invented tenth class is refused — never guessed downstream.
        ReplyClassification(
            reply_class="spam", confidence=0.8,
            rationale="The reply is unsolicited bulk mail that is not a real answer.",
            evidence_quote="BUY NOW limited time offer",
        )
    with pytest.raises(ValidationError):
        # Confidence outside [0,1] is nonsensical, not silently clamped.
        ReplyClassification(
            reply_class="positive", confidence=1.5,
            rationale="The reply asks for more detail about the offer.",
            evidence_quote="Could you send over a bit more detail",
        )
    with pytest.raises(ValidationError):
        # A token rationale proves nothing — the floor forces an argument.
        ReplyClassification(
            reply_class="positive", confidence=0.8,
            rationale="seems positive",
            evidence_quote="Could you send over a bit more detail",
        )


def test_policy_gate_decision_decision_constrained():
    decision = PolicyGateDecision(
        action="score_lead",
        decision="allow",
        reasons=["all required fields present"],
        matched_rules=["P4"],
        required_fields_missing=[],
        risk_level="low",
    )
    assert decision.decision == "allow"
    with pytest.raises(ValidationError):
        PolicyGateDecision(
            action="score_lead", decision="maybe", reasons=[], matched_rules=[],
            required_fields_missing=[], risk_level="low",
        )
