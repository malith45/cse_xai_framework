"""Tests for the CSE security identifier normaliser (src/identifiers)."""

import math

import pytest

from identifiers import (
    KNOWN_SHARE_CLASSES,
    MalformedIdentifier,
    SecurityId,
    UnknownShareClass,
    UnresolvedSecurityId,
    parse,
    parse_bare,
    parse_components,
    parse_dot,
    parse_hyphen,
    to_bare,
    to_components,
    to_dot,
    to_hyphen,
)

AAF_N = SecurityId("AAF", "N", "0000")


# --------------------------------------------------------------------------- canonical form
class TestCanonicalForm:
    def test_canonical_string_is_hyphen_form(self):
        assert AAF_N.canonical == "AAF-N-0000"
        assert str(AAF_N) == "AAF-N-0000"

    def test_is_hashable_and_value_equal(self):
        assert SecurityId("AAF", "N", "0") == AAF_N
        assert hash(SecurityId("AAF", "N", "0")) == hash(AAF_N)
        assert {AAF_N: 1}[SecurityId("aaf", "n", 0)] == 1

    def test_is_immutable(self):
        with pytest.raises(AttributeError):
            AAF_N.share_class = "X"  # type: ignore[misc]

    @pytest.mark.parametrize("sub", ["0000", "0", 0, 0.0, "0.0", " 0 "])
    def test_sub_type_zero_variants_normalise(self, sub):
        assert SecurityId("AAF", "N", sub).sub_type == "0000"

    @pytest.mark.parametrize("sub,expected", [("1", "0001"), (1, "0001"), (1.0, "0001"), ("0024", "0024"), ("5", "0005")])
    def test_sub_type_nonzero_preserved_and_padded(self, sub, expected):
        assert SecurityId("AAF", "W", sub).sub_type == expected

    def test_all_observed_share_classes_accepted(self):
        for cls in KNOWN_SHARE_CLASSES:
            assert SecurityId("AAF", cls).share_class == cls


# --------------------------------------------------------------------------- round trips
class TestRoundTrip:
    def test_components_round_trip(self):
        sid = parse_components("AAF", "N", "0000")
        assert to_components(sid) == ("AAF", "N", "0000")
        assert parse_components(*to_components(sid)) == sid

    def test_components_round_trip_with_numeric_sub_type(self):
        # pandas reads SUB TYPE as float in the 2022 Q3/Q4, 2023 and 2025 files
        sid = parse_components("AAF", "N", 0.0)
        assert to_components(sid) == ("AAF", "N", "0000")

    def test_hyphen_round_trip(self):
        sid = parse_hyphen("AAF-N-0000")
        assert to_hyphen(sid) == "AAF-N-0000"
        assert parse_hyphen(to_hyphen(sid)) == sid

    def test_dot_round_trip(self):
        sid = parse_dot("AAF.N0000")
        assert to_dot(sid) == "AAF.N0000"
        assert parse_dot(to_dot(sid)) == sid

    def test_bare_round_trip_via_unresolved(self):
        u = parse_bare("JKH")
        assert to_bare(u) == "JKH"
        assert parse_bare(to_bare(u)) == u

    def test_bare_from_resolved_is_lossy_but_stable(self):
        assert to_bare(SecurityId("JKH", "N")) == "JKH"

    @pytest.mark.parametrize(
        "sid",
        [SecurityId("COMB", "X", "0000"), SecurityId("CDB", "P", "0002"), SecurityId("HNB", "W", "0024"), SecurityId("NAMU", "U")],
    )
    def test_cross_format_round_trip_preserves_class_and_sub_type(self, sid):
        assert parse_hyphen(to_hyphen(sid)) == sid
        assert parse_dot(to_dot(sid)) == sid
        assert parse_components(*to_components(sid)) == sid
        assert parse(to_hyphen(sid)) == parse(to_dot(sid)) == parse(to_components(sid)) == sid


# --------------------------------------------------------------------------- auto-detection
class TestAutoDetect:
    def test_detects_dot(self):
        assert parse("AAF.N0000") == AAF_N

    def test_detects_hyphen(self):
        assert parse("AAF-N-0000") == AAF_N

    def test_detects_components_tuple(self):
        assert parse(("AAF", "N", "0")) == AAF_N

    def test_detects_components_list(self):
        assert parse(["AAF", "N", 0.0]) == AAF_N

    def test_detects_bare_as_unresolved(self):
        result = parse("AAF")
        assert isinstance(result, UnresolvedSecurityId)
        assert result.company_code == "AAF"

    def test_three_stated_formats_give_the_same_canonical_id(self):
        # The author's acceptance check: AAF + N + 0, AAF-N-0000 and AAF.N0000 are one security.
        a = parse(("AAF", "N", "0"))
        b = parse("AAF-N-0000")
        c = parse("AAF.N0000")
        assert a == b == c
        assert a.canonical == b.canonical == c.canonical == "AAF-N-0000"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # Real strings observed in the CSE Data Library
            ("AEL-N-0000", "AEL-N-0000"),          # Dividends 2025
            ("CDB-X-0000", "CDB-X-0000"),          # Dividends 2025
            ("TILE-N-0", "TILE-N-0000"),           # Dividends 2010 (one-digit sub type)
            ("AAF -R-0000", "AAF-R-0000"),         # Quoted securities 2014 (stray space)
            ("AAF/BD/18/12/27", None),             # Quoted securities: debenture, must be rejected
            ("AAIC.N0000", "AAIC-N-0000"),         # Public holding 2025
            ("VLL.x0000", "VLL-X-0000"),           # Public holding: lower-case class
            ("SPEN.N0000", "SPEN-N-0000"),         # Block trades 2025
            ("SUN.N0000", "SUN-N-0000"),           # Splits 2025
            ("JKH", "unresolved"),                 # Splits 2024 (bare)
            ("Unisyst", None),                     # Splits 2021: a company name, must be rejected
            ("COMB-N-0000 & COMB-X-0000", None),   # Dividends: joint entry, must be rejected
        ],
    )
    def test_real_world_strings(self, raw, expected):
        if expected is None:
            with pytest.raises(MalformedIdentifier):
                parse(raw)
        elif expected == "unresolved":
            assert isinstance(parse(raw), UnresolvedSecurityId)
        else:
            assert parse(raw).canonical == expected


# --------------------------------------------------------------------------- unknown class
class TestUnknownClass:
    def test_bare_ticker_is_not_guessed(self):
        u = parse("JKH")
        assert isinstance(u, UnresolvedSecurityId)
        assert u.is_resolved is False
        assert u.canonical is None
        assert not isinstance(u, SecurityId)
        assert u.raw == "JKH"

    def test_blank_main_type_in_components_is_unresolved(self):
        assert isinstance(parse_components("JKH", "", "0000"), UnresolvedSecurityId)
        assert isinstance(parse_components("JKH", None, "0000"), UnresolvedSecurityId)
        assert isinstance(parse_components("JKH", math.nan, "0000"), UnresolvedSecurityId)

    def test_formatters_refuse_unresolved(self):
        u = parse("JKH")
        for fmt in (to_hyphen, to_dot, to_components):
            with pytest.raises(TypeError):
                fmt(u)  # type: ignore[arg-type]
        assert to_bare(u) == "JKH"

    def test_caller_can_resolve(self):
        u = parse("JKH")
        assert u.resolve("N") == SecurityId("JKH", "N", "0000")
        assert u.resolve("X", "0001") == SecurityId("JKH", "X", "0001")

    def test_resolve_still_validates_class(self):
        with pytest.raises(UnknownShareClass):
            parse("JKH").resolve("Z")

    def test_unresolved_never_equals_resolved(self):
        assert parse("JKH") != SecurityId("JKH", "N")
        assert parse("JKH") != SecurityId("JKH", "X")

    def test_unresolved_str_marks_unknown(self):
        assert str(parse("JKH")) == "JKH-?-????"


# --------------------------------------------------------------------------- malformed input
class TestMalformed:
    @pytest.mark.parametrize(
        "raw",
        [
            "", "   ", None, 12345, 1.5, True, b"AAF-N-0000",
            "AAF/BD/18/12/27",            # debenture code
            "COMB-N-0000 & COMB-X-0000",  # joint entry
            "AAF-N-ABCD",                 # non-numeric sub type
            "AAF-N-00000",                # five-digit sub type
            "AAF.N00000",
            "AAF-NN-0000",                # two-letter class
            "AAF.0000",                   # missing class
            "AAF-N",                      # missing sub type
            "-N-0000",                    # missing code
            "A.N0000",                    # one-character code
            "TOOLONGCODE.N0000",
            "Unisyst",                    # company name
            "AAF N 0000",                 # spaces instead of delimiters
            "AAF_N_0000",
        ],
    )
    def test_parse_rejects(self, raw):
        with pytest.raises(MalformedIdentifier):
            parse(raw)

    def test_unknown_share_class_letter(self):
        with pytest.raises(UnknownShareClass):
            parse("AAF-Z-0000")
        with pytest.raises(UnknownShareClass):
            parse("AAF.Q0000")
        with pytest.raises(UnknownShareClass):
            parse_components("AAF", "Z", "0000")

    def test_unknown_share_class_is_a_malformed_identifier(self):
        assert issubclass(UnknownShareClass, MalformedIdentifier)

    @pytest.mark.parametrize("bad", [(), ("AAF",), ("AAF", "N"), ("AAF", "N", "0000", "extra")])
    def test_components_wrong_arity(self, bad):
        with pytest.raises(MalformedIdentifier):
            parse(bad)

    @pytest.mark.parametrize("sub", ["", None, math.nan, "abcd", "-1", 1.5, 10000, "12345"])
    def test_components_bad_sub_type(self, sub):
        with pytest.raises(MalformedIdentifier):
            parse_components("AAF", "N", sub)

    def test_specific_parsers_reject_other_formats(self):
        with pytest.raises(MalformedIdentifier):
            parse_hyphen("AAF.N0000")
        with pytest.raises(MalformedIdentifier):
            parse_dot("AAF-N-0000")
        with pytest.raises(MalformedIdentifier):
            parse_bare("AAF-N-0000")


# --------------------------------------------------------------------------- whitespace & case
class TestWhitespaceAndCase:
    @pytest.mark.parametrize(
        "raw",
        ["aaf.n0000", "  AAF.N0000  ", "Aaf.N0000", "AAF .N0000", "\tAAF.N0000\n"],
    )
    def test_dot_variants(self, raw):
        assert parse(raw) == AAF_N
        assert parse_dot(raw) == AAF_N

    @pytest.mark.parametrize(
        "raw",
        ["aaf-n-0000", "  AAF-N-0000  ", "AAF -N-0000", "AAF - N - 0000", "aaf-N-0"],
    )
    def test_hyphen_variants(self, raw):
        assert parse(raw) == AAF_N
        assert parse_hyphen(raw) == AAF_N

    @pytest.mark.parametrize("raw", ["aaf", "  AAF ", "Aaf"])
    def test_bare_variants(self, raw):
        u = parse(raw)
        assert isinstance(u, UnresolvedSecurityId)
        assert u.company_code == "AAF"

    @pytest.mark.parametrize(
        "code,cls,sub",
        [(" AAF ", " N ", " 0000 "), ("aaf", "n", "0"), ("Aaf", "n", 0.0), ("AAF", "N ", "0000")],
    )
    def test_components_variants(self, code, cls, sub):
        assert parse_components(code, cls, sub) == AAF_N

    def test_output_is_always_upper_case_without_whitespace(self):
        sid = parse("  aaf.n0000 ")
        assert to_hyphen(sid) == "AAF-N-0000"
        assert to_dot(sid) == "AAF.N0000"
        assert to_components(sid) == ("AAF", "N", "0000")
        assert to_bare(sid) == "AAF"


# --------------------------------------------------------------------------- share classes distinct
class TestShareClassesNeverCollapse:
    def test_comb_voting_and_non_voting_are_different_securities(self):
        voting = parse("COMB.N0000")
        non_voting = parse("COMB.X0000")
        assert voting != non_voting
        assert voting.canonical != non_voting.canonical
        assert hash(voting) != hash(non_voting)
        assert voting.canonical == "COMB-N-0000"
        assert non_voting.canonical == "COMB-X-0000"
        assert len({voting, non_voting}) == 2

    def test_same_across_every_input_format(self):
        n_forms = [parse("COMB.N0000"), parse("COMB-N-0000"), parse(("COMB", "N", "0"))]
        x_forms = [parse("COMB.X0000"), parse("COMB-X-0000"), parse(("COMB", "X", "0"))]
        assert len(set(n_forms)) == 1
        assert len(set(x_forms)) == 1
        assert set(n_forms).isdisjoint(set(x_forms))

    def test_preference_is_distinct_too(self):
        assert len({parse("CDB.N0000"), parse("CDB.X0000"), parse("CDB.P0000")}) == 3

    def test_sub_type_also_distinguishes(self):
        # Preference share tranches (P-0001 ... P-0005) are separate securities
        assert parse("CDB-P-0001") != parse("CDB-P-0002")

    def test_to_bare_is_the_only_lossy_formatter(self):
        assert to_bare(parse("COMB.N0000")) == to_bare(parse("COMB.X0000")) == "COMB"
        assert to_dot(parse("COMB.N0000")) != to_dot(parse("COMB.X0000"))
        assert to_hyphen(parse("COMB.N0000")) != to_hyphen(parse("COMB.X0000"))
        assert to_components(parse("COMB.N0000")) != to_components(parse("COMB.X0000"))
