"""M3 helpers that need no model: Unicode-script language routing, entity linking with the trilingual
alias dictionary (decision D6), and trading-day assignment with the 14:30 cut-off (§4.6.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

SINHALA = re.compile(r"[඀-෿]")
TAMIL = re.compile(r"[஀-௿]")
LATIN = re.compile(r"[A-Za-z]")


def script_language(text: str) -> str:
    """'si' / 'ta' when that script dominates the letters; 'latin' otherwise (fastText then decides
    between English and code-mixed); 'unknown' for text without letters."""
    s, t, l = len(SINHALA.findall(text)), len(TAMIL.findall(text)), len(LATIN.findall(text))
    if s + t + l == 0:
        return "unknown"
    if s >= max(t, l):
        return "si"
    if t >= max(s, l):
        return "ta"
    return "latin"


# --------------------------------------------------------------------------- alias dictionary (D6)
# Tickers, English names, Sinhala and Tamil names and key brands. Sinhala/Tamil spellings are the
# common news-media spellings and MUST be checked by a native reader before results are reported.
ALIASES: dict[str, list[str]] = {
    "JKH-N-0000": ["John Keells", "JKH", "Keells", "ජෝන් කීල්ස්", "කීල්ස්", "ஜோன் கீல்ஸ்", "கீல்ஸ்"],
    "COMB-N-0000": ["Commercial Bank", "ComBank", "COMB", "කොමර්ෂල් බැංකුව", "කොමර්ෂල්", "கொமர்ஷல் வங்கி"],
    "HNB-N-0000": ["Hatton National Bank", "HNB", "හැටන් නැෂනල් බැංකුව", "හැටන් නැෂනල්", "ஹட்டன் நெஷனல் வங்கி"],
    "SAMP-N-0000": ["Sampath Bank", "SAMP", "සම්පත් බැංකුව", "සම්පත් බැංකු", "சம்பத் வங்கி"],
    "DIAL-N-0000": ["Dialog Axiata", "Dialog", "DIAL", "ඩයලොග්", "டயலொக்"],
    "LOLC-N-0000": ["LOLC Holdings", "LOLC", "එල්ඕඑල්සී", "எல்.ஓ.எல்.சி"],
    "HAYL-N-0000": ["Hayleys", "HAYL", "හේලීස්", "ஹேலீஸ்"],
    "CTC-N-0000": ["Ceylon Tobacco", "CTC", "ලංකා දුම්කොළ සමාගම", "සිලෝන් ටොබැකෝ", "இலங்கை புகையிலை நிறுவன"],
    "LIOC-N-0000": ["Lanka IOC", "LIOC", "ලංකා අයිඕසී", "ලංකා IOC", "லங்கா ஐஓசி", "லங்கா IOC"],
    "MELS-N-0000": ["Melstacorp", "MELS", "මෙල්ස්ටා", "மெல்ஸ்டா"],
    "VONE-N-0000": ["Vallibel One", "VONE", "වැලිබෙල් වන්", "வலிபெல் வன்"],
    "NDB-N-0000": ["National Development Bank", "NDB Bank", "NDB", "ජාතික සංවර්ධන බැංකුව", "தேசிய அபிவிருத்தி வங்கி"],
    "DFCC-N-0000": ["DFCC Bank", "DFCC", "ඩීඑෆ්සීසී", "டிஎவ்சிசி"],
    "HHL-N-0000": ["Hemas Holdings", "Hemas", "HHL", "හේමාස්", "ஹேமாஸ்"],
    "SUN-N-0000": ["Sunshine Holdings", "SUN", "සන්ෂයින්", "சன்ஷைன்"],
    "CARG-N-0000": ["Cargills", "CARG", "කාගිල්ස්", "கார்கில்ஸ்"],
    "CINS-N-0000": ["Ceylinco Insurance", "CINS", "සෙලින්කෝ රක්ෂණ", "செலிங்கோ காப்புறுதி"],
    "CIC-N-0000": ["CIC Holdings", "CIC", "සීඅයිසී", "சிஐசி"],
    "TJL-N-0000": ["Teejay Lanka", "Teejay", "TJL", "ටීජේ ලංකා", "டீஜே லங்கா"],
    "LLUB-N-0000": ["Chevron Lubricants", "LLUB", "ෂෙව්රොන්", "செவ்ரோன்"],
    "TKYO-N-0000": ["Tokyo Cement", "TKYO", "ටෝකියෝ සිමෙන්ති", "டோக்கியோ சீமெந்து"],
    "AEL-N-0000": ["Access Engineering", "AEL", "ඇක්සස් ඉංජිනේරු", "அக்சஸ் இன்ஜினியரிங்"],
    "RCL-N-0000": ["Royal Ceramics", "RCL", "රෝයල් සෙරමික්", "ரோயல் செரமிக்"],
    "DIPD-N-0000": ["Dipped Products", "DIPD", "ඩිප්ඩ් ප්‍රොඩක්ට්ස්", "டிப்ட் புரொடக்ட்ஸ்"],
    "EXPO-N-0000": ["Expolanka", "EXPO", "එක්ස්පෝලංකා", "எக்ஸ்போலங்கா"],
    "BIL-N-0000": ["Browns Investments", "BIL", "බ්‍රවුන්ස් ඉන්වෙස්ට්මන්ට්ස්", "பிரவுன்ஸ் இன்வெஸ்ட்மன்ட்ஸ்"],
    "AAIC-N-0000": ["Softlogic Life", "AAIC", "සොෆ්ට්ලොජික් ලයිෆ්", "சொஃப்ட்லொஜிக் லைஃப்"],
    "LOFC-N-0000": ["LOLC Finance", "LOFC", "එල්ඕඑල්සී ෆිනෑන්ස්", "எல்.ஓ.எல்.சி பினான்ஸ்"],
    "CCS-N-0000": ["Ceylon Cold Stores", "Elephant House", "CCS", "එලිෆන්ට් හවුස්", "எலிபன்ட் ஹவுஸ்"],
    "LION-N-0000": ["Lion Brewery", "LION", "ලයන් බ්‍රිවරි", "லயன் புரூவரி"],
    "SPEN-N-0000": ["Aitken Spence", "SPEN", "එයිට්කන් ස්පෙන්ස්", "எயிட்கன் ஸ்பென்ஸ்"],
    "SLTL-N-0000": ["Sri Lanka Telecom", "SLT-Mobitel", "SLTL", "ශ්‍රී ලංකා ටෙලිකොම්", "ஸ்ரீலங்கா டெலிகொம்"],
    "PLC-N-0000": ["People's Leasing", "PLC", "පීපල්ස් ලීසිං", "பீப்பள்ஸ் லீசிங்"],
    "LFIN-N-0000": ["LB Finance", "LFIN", "එල්බී ෆිනෑන්ස්", "எல்பி பினான்ஸ்"],
    "CALT-N-0000": ["Capital Alliance", "CALT", "කැපිටල් අලයන්ස්", "கெப்பிட்டல் அலையன்ஸ்"],
    "MGT-N-0000": ["Hayleys Fabric", "MGT", "හේලීස් ෆැබ්‍රික්", "ஹேலீஸ் ஃபேப்ரிக்"],
    "WATA-N-0000": ["Watawala Plantations", "WATA", "වටවල වැවිලි", "வட்டவளை பெருந்தோட்ட"],
    "LWL-N-0000": ["Lanka Walltiles", "LWL", "ලංකා වෝල්ටයිල්ස්", "லங்கா வோல்டைல்ஸ்"],
    "CINV-N-0000": ["Ceylon Investment", "CINV"],
    "PKME-N-0000": ["PickMe", "PKME", "පික්මී", "பிக்மீ"],
}
# Short or ambiguous aliases only count as a match when written as a whole word in upper case.
STRICT = {"CTC", "SUN", "CIC", "PLC", "AEL", "RCL", "MGT", "LWL", "BIL", "CCS", "HNB", "NDB", "JKH", "SLTL"}

MARKET_TERMS = ["ASPI", "S&P SL20", "Colombo Stock Exchange", "CSE", "stock market", "share market",
                # Sinhala (both spellings of "market" occur in Hiru text): stock market, Colombo stock
                # (exchange/market), share exchange, All Share Price Index
                "කොටස් වෙළෙඳපොළ", "කොටස් වෙළඳපොළ", "කොළඹ කොටස්", "කොටස් හුවමාරු", "පොදු කොටස් මිල දර්ශකය",
                "සියලු කොටස් මිල දර්ශකය",
                # Tamil: share market (with and without the space), Colombo share exchange, All Share Price Index
                "பங்குச் சந்தை", "பங்குச்சந்தை", "கொழும்பு பங்குச் சந்தை", "கொழும்பு பங்குப் பரிவர்த்தனை",
                "அனைத்து பங்கு விலைச் சுட்டெண்"]


def _pattern(alias: str) -> re.Pattern:
    if alias in STRICT:
        return re.compile(rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])")
    if LATIN.search(alias):
        return re.compile(rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])", re.IGNORECASE)
    return re.compile(re.escape(alias))


PATTERNS = {sid: [_pattern(a) for a in al] for sid, al in ALIASES.items()}
MARKET_PATTERNS = [_pattern(a) for a in MARKET_TERMS]


def link_companies(text: str) -> list[str]:
    return [sid for sid, pats in PATTERNS.items() if any(p.search(text) for p in pats)]


def is_market_news(text: str) -> bool:
    return any(p.search(text) for p in MARKET_PATTERNS)


# --------------------------------------------------------------------------- trading-day assignment

CUTOFF = pd.Timedelta(hours=14, minutes=30)   # CSE regular session close, Sri Lanka time


def assign_trading_day(published: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """News published at or before 14:30 on a trading day counts for that day's close; news after
    14:30, on a non-trading day, or with no time of day (conservatively treated as after the close)
    counts for the next trading day."""
    def local(v):
        t = pd.to_datetime(v, errors="coerce")
        if t is pd.NaT or pd.isna(t):
            return pd.NaT
        return t.tz_convert("Asia/Colombo").tz_localize(None) if t.tzinfo else t   # naive = Sri Lanka time

    ts = pd.to_datetime(published.map(local))
    has_time = published.astype(str).str.contains(r"\d{1,2}:\d{2}", regex=True)
    day = ts.dt.normalize()
    after = (~has_time) | ((ts - day) > CUTOFF)
    cal = calendar.sort_values()
    pos = cal.searchsorted(day.values)            # first trading day >= day
    on_cal = (pos < len(cal)) & (cal[np.minimum(pos, len(cal) - 1)] == day.values)
    nxt = np.where(on_cal & ~after.values, pos, np.where(on_cal, pos + 1, pos))
    out = pd.Series(pd.NaT, index=published.index, dtype="datetime64[ns]")
    ok = (nxt < len(cal)) & ts.notna().values
    out[ok] = cal[nxt[ok]]
    return out
