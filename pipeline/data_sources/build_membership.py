"""Point-in-time S&P SL20 membership (ordinary voting shares, N class) from the CSE rebalancing
announcements, 2020-2025. Each official CSE PDF from June 2020 onwards prints the FULL constituent list
after the change, so every list below is copied from its PDF, except: the base list (derived from the
June 2020 PDF: list minus inclusions plus exclusions), December 2020, June 2021 and March 2022, which
are derived from the previous list plus the changes reported in national newspapers that reproduce the
CSE media release (the CSE no longer hosts those PDFs). Every transition is checked below: the stated
inclusions/exclusions must turn each list into the next one exactly.
"""
import pandas as pd

CSE = "https://cdn.cse.lk/"
LISTS = [
 ("2020-06-22", "CSE PDF 19 Jun 2020", CSE + "pdf/press/Constituent-Changes-to-the-S&P-Sri-Lanka-20-Index.pdf",
  "AEL SPEN CFIN CTC LLUB COMB DFCC DIAL HNB JKH LOFC LOLC MELS NDB OSEA PLC RICH SAMP SEYB TJL"),
 ("2021-12-20", "CSE PDF 17 Dec 2021", CSE + "pdf/company-articles/Constituent-Changes-to-the-SP-Sri-Lanka-20-Index.pdf",
  "LLUB COMB CLC DIAL DIPD EXPO HNB HAYL HHL JKH LFIN LOFC LOLC MELS NDB RCL SAMP TJL TKYO VONE"),
 ("2022-06-20", "CSE PDF 17 Jun 2022", CSE + "cms-internal/news/U7g8qBV2SH1nL5Ny_17Jun2022084244GMT_1655455364775.pdf",
  "LLUB COMB DIAL DIPD EXPO HAYL HELA HHL JKH LIOC LFIN LOFC LOLC MELS RCL SAMP AAIC TJL VONE WATA"),
 ("2022-12-19", "CSE PDF 16 Dec 2022", CSE + "cms-internal/news/D1h29zZ47vb44wE1_16Dec2022111849GMT_1671189529913.pdf",
  "BRWN BIL LLUB CIC COMB DIPD EXPO HAYL JKH LIOC LOFC LOLC MELS RICH RCL SAMP AAIC TJL VONE WATA"),
 ("2023-06-19", "CSE PDF 16 Jun 2023", CSE + "cms-internal/news/BDFHJNkXIcP2qcsL_16Jun2023103934GMT_1686911974122.pdf",
  "BIL LLUB CIC COMB DIPD EXPO HAYL HNB HHL JKH LIOC LOFC LOLC MELS RICH RCL SAMP AAIC SLTL VONE"),
 ("2023-12-18", "CSE PDF 15 Dec 2023", CSE + "cms-internal/NEWS/ZWqgF7M1jwGGMiTR_18Dec2023074241GMT_1702885361521.pdf",
  "LLUB CIC COMB DFCC DIAL DIST HNB HAYL HHL JKH LIOC LOFC LOLC MELS NDB NTB RCL SAMP AAIC VONE"),
 ("2024-06-24", "CSE PDF 24 Jun 2024", CSE + "cms-internal/news/HPIY5MnKx3H4hDQ0_25Jun2024040756GMT_1719288476025.pdf",
  "LLUB CIC COMB DFCC DIAL DIST HNB HAYL HHL JKH LIOC LFIN LOFC LOLC MELS NDB NTB RCL SAMP VONE"),
 ("2024-12-23", "CSE PDF 20 Dec 2024", CSE + "cms-internal/news/Hs6n2VdFDacA0JRG_20Dec2024134651GMT_1734702411651.pdf",
  "CCS LLUB COMB DFCC DIAL DIST HNB HAYL HHL JKH LIOC LFIN LOFC LOLC MELS NDB NTB RCL SAMP VONE"),
 ("2025-06-23", "CSE PDF 20 Jun 2025", CSE + "cms-internal/news/GRVsu9KIz2h9lvwo_23Jun2025024621GMT_1750646781308.pdf",
  "CFIN CCS LLUB COMB DFCC DIAL HNB HAYL HHL JKH LIOC LFIN LOFC LOLC MELS NDB NTB RCL SAMP VONE"),
 ("2025-12-22", "CSE PDF 19 Dec 2025", CSE + "cms-internal/NEWS/o6yXCXeEqGEsETzZ_22Dec2025060353GMT_1766383433648.pdf",
  "AEL CFIN COMB DFCC DIAL HNB HAYL HHL JKH LIOC LFIN LOLC MELS NDB NTB PLC SAMP SINS SUN VONE"),
]
# (effective date, source, url, added, deleted) - N-class securities only
CHANGES = [
 ("2020-06-22", "CSE PDF 19 Jun 2020", LISTS[0][2], "CFIN CTC LOLC MELS RICH SEYB", "TKYO EXPO JINS LIOC AAIC UBC"),
 ("2020-12-21", "Sunday Observer 27 Dec 2020 (reproduces CSE release of 18 Dec 2020)", "https://archives1.sundayobserver.lk/node/42905",
  "DIPD DIST EXPO HAYL HHL RCL TKYO", "SPEN CFIN CTC LOFC OSEA RICH SEYB"),
 ("2021-06-21", "Daily FT 21 Jun 2021 (reproduces CSE release of 18 Jun 2021)", "https://www.ft.lk/Financial-Services/CSE-makes-constituent-changes-to-S-P-Sri-Lanka-20-index/42-719442",
  "BRWN LFIN RICH VONE", "AEL DFCC DIST PLC"),   # Commercial Bank removal was the non-voting COMB.X only
 ("2021-12-20", LISTS[1][1], LISTS[1][2], "CLC LOFC", "RICH BRWN"),
 ("2022-03-31", "Sunday Observer 3 Apr 2022 (ad hoc; CLC merged into LOFC)", "https://archives1.sundayobserver.lk/node/55652", "AAIC", "CLC"),
 ("2022-06-20", LISTS[2][1], LISTS[2][2], "LIOC WATA HELA", "TKYO NDB HNB"),
 ("2022-12-19", LISTS[3][1], LISTS[3][2], "BIL BRWN CIC RICH", "DIAL HELA HHL LFIN"),
 ("2023-06-19", LISTS[4][1], LISTS[4][2], "HNB HHL SLTL", "BRWN TJL WATA"),
 ("2023-12-18", LISTS[5][1], LISTS[5][2], "DFCC DIAL DIST NTB NDB", "BIL DIPD EXPO RICH SLTL"),
 ("2024-06-24", LISTS[6][1], LISTS[6][2], "LFIN", "AAIC"),
 ("2024-12-23", LISTS[7][1], LISTS[7][2], "CCS", "CIC"),
 ("2025-06-23", LISTS[8][1], LISTS[8][2], "CFIN", "DIST"),
 ("2025-12-22", LISTS[9][1], LISTS[9][2], "AEL PLC SINS SUN", "CCS LLUB LOFC RCL"),
]


def main():
    s = lambda x: set(x.split())
    # base list in force on 2 Jan 2020 = June 2020 list - inclusions + exclusions
    base = (s(LISTS[0][3]) - s(CHANGES[0][3])) | s(CHANGES[0][4])
    assert len(base) == 20
    snaps = [("2020-01-02", "Derived from CSE PDF 19 Jun 2020 (list - inclusions + exclusions)", LISTS[0][2], base)]
    cur = base
    full = {d: s(m) for d, _, _, m in LISTS}
    for d, src, url, add, dele in CHANGES:
        assert s(dele) <= cur, (d, s(dele) - cur)
        assert not (s(add) & cur), (d, s(add) & cur)
        cur = (cur - s(dele)) | s(add)
        assert len(cur) == 20, (d, len(cur))
        if d in full:
            assert cur == full[d], (d, "mismatch with printed full list", cur ^ full[d])
        snaps.append((d, src, url, cur))
    rows = []
    for i, (d, src, url, members) in enumerate(snaps):
        end = (pd.Timestamp(snaps[i + 1][0]) - pd.Timedelta(days=1)).date().isoformat() if i + 1 < len(snaps) else ""
        for code in sorted(members):
            rows.append({"security_id": f"{code}-N-0000", "start_date": d, "end_date": end, "source": src, "source_url": url})
    df = pd.DataFrame(rows)
    # merge consecutive spells of the same security
    df = df.sort_values(["security_id", "start_date"])
    out = []
    for sid, g in df.groupby("security_id"):
        spell = None
        for r in g.itertuples():
            if spell and spell["end_date"] and (pd.Timestamp(spell["end_date"]) + pd.Timedelta(days=1)) == pd.Timestamp(r.start_date):
                spell["end_date"] = r.end_date
                spell["source"] += "; " + r.source
            else:
                if spell:
                    out.append(spell)
                spell = {"security_id": sid, "start_date": r.start_date, "end_date": r.end_date, "source": r.source}
        out.append(spell)
    return pd.DataFrame(out), snaps


if __name__ == "__main__":
    import sys
    m, snaps = main()
    m.to_csv(sys.argv[1] if len(sys.argv) > 1 else "membership.csv", index=False)
    print(f"{len(snaps)} snapshots, all transitions consistent; {m['security_id'].nunique()} securities, {len(m)} spells")
