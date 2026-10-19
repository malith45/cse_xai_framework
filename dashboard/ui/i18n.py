"""Interface language (English / Sinhala / Tamil).

Scope (author's decision 2026-10-20): menu, page titles, section headings, buttons, card labels,
signal labels, disclaimer and the short "How to use this page" guides. Chart notes, data tables and
the generated plain-language explanations stay in English. Sinhala and Tamil wording is a first
draft to be checked by native speakers.

tr(text) returns the translation of an exact English UI string, or the English text if none exists.
"""
from __future__ import annotations

import streamlit as st

LANGS = {"en": "English", "si": "සිංහල", "ta": "தமிழ்"}

T: dict[str, tuple[str, str]] = {   # English -> (Sinhala, Tamil)
    # navigation / pages
    "Market overview": ("වෙළඳපොළ සාරාංශය", "சந்தை கண்ணோட்டம்"),
    "Stock prediction": ("කොටස් පුරෝකථනය", "பங்கு முன்னறிவிப்பு"),
    "Explainability": ("පැහැදිලි කිරීම", "விளக்கம்"),
    "Risk & stress test": ("අවදානම සහ ආතති පරීක්ෂාව", "இடர் மற்றும் அழுத்த சோதனை"),
    "News sentiment": ("පුවත් මනෝභාවය", "செய்தி உணர்வு"),
    # sidebar
    "Language": ("භාෂාව", "மொழி"),
    "Selection": ("තේරීම", "தேர்வு"),
    "Stock": ("කොටස", "பங்கு"),
    "As-of date": ("දිනය", "திகதி"),
    "Horizon": ("කාල සීමාව", "காலம்"),
    "Prediction horizon": ("පුරෝකථන කාල සීමාව", "முன்னறிவிப்புக் காலம்"),
    "Next trading day (t+1)": ("ඊළඟ වෙළඳ දිනය (t+1)", "அடுத்த வர்த்தக நாள் (t+1)"),
    "Five trading days (t+5)": ("වෙළඳ දින පහ (t+5)", "ஐந்து வர்த்தக நாட்கள் (t+5)"),
    "Open evaluation questionnaire": ("ඇගයීම් ප්‍රශ්නාවලිය විවෘත කරන්න", "மதிப்பீட்டு கேள்வித்தாளைத் திறக்கவும்"),
    "Sign out": ("ඉවත් වන්න", "வெளியேறு"),
    "Data as of": ("දත්ත දිනය", "தரவு திகதி"),
    "Source": ("මූලාශ්‍රය", "மூலம்"),
    "Model": ("ආකෘතිය", "மாதிரி"),
    "How to use this page": ("මෙම පිටුව භාවිත කරන ආකාරය", "இந்தப் பக்கத்தைப் பயன்படுத்தும் முறை"),
    "About this chart": ("මෙම ප්‍රස්ථාරය ගැන", "இந்த வரைபடம் பற்றி"),
    "Details for experts": ("විශේෂඥ විස්තර", "நிபுணர்களுக்கான விவரங்கள்"),
    # sections
    "Index history": ("දර්ශක ඉතිහාසය", "சுட்டெண் வரலாறு"),
    "Market sentiment": ("වෙළඳපොළ මනෝභාවය", "சந்தை உணர்வு"),
    "Top movers": ("වැඩිපුරම වෙනස් වූ කොටස්", "அதிகம் மாறிய பங்குகள்"),
    "Recent price": ("මෑත මිල", "அண்மைய விலை"),
    "What drives predictions overall": ("පුරෝකථන වලට සමස්තයක් ලෙස බලපාන සාධක", "முன்னறிவிப்புகளை பொதுவாக இயக்கும் காரணிகள்"),
    "What drove this prediction": ("මෙම පුරෝකථනයට බලපෑ සාධක", "இந்த முன்னறிவிப்பை இயக்கிய காரணிகள்"),
    "In plain language": ("සරල භාෂාවෙන්", "எளிய மொழியில்"),
    "Scenario": ("අවස්ථාව", "சூழ்நிலை"),
    "Value at risk": ("අවදානම් අගය (VaR)", "இடர் மதிப்பு (VaR)"),
    "Sentiment signal": ("මනෝභාව සංඥාව", "உணர்வு சமிக்ஞை"),
    "Articles": ("ලිපි", "கட்டுரைகள்"),
    # cards / labels
    "Breadth": ("වෙළඳපොළ පුළුල", "சந்தை அகலம்"),
    "Market risk": ("වෙළඳපොළ අවදානම", "சந்தை இடர்"),
    "Calm": ("සන්සුන්", "அமைதி"), "Moderate": ("මධ්‍යස්ථ", "மிதமானது"), "Elevated": ("ඉහළ", "உயர்ந்தது"),
    "Next day (t+1)": ("ඊළඟ දිනය (t+1)", "அடுத்த நாள் (t+1)"),
    "Five days (t+5)": ("දින පහ (t+5)", "ஐந்து நாட்கள் (t+5)"),
    "Up": ("ඉහළට", "உயர்வு"), "Down": ("පහළට", "இறக்கம்"), "Unclear": ("අපැහැදිලියි", "தெளிவில்லை"),
    "strong signal": ("ශක්තිමත් සංඥාව", "வலுவான சமிக்ஞை"),
    "moderate signal": ("මධ්‍යස්ථ සංඥාව", "மிதமான சமிக்ஞை"),
    "weak evidence at this horizon": ("මෙම කාල සීමාවට සාක්ෂි දුර්වලයි", "இந்தக் காலத்துக்கு சான்று பலவீனம்"),
    "Position size (LKR)": ("ආයෝජන ප්‍රමාණය (රු.)", "முதலீட்டு அளவு (ரூ.)"),
    "Volatility multiplier": ("උච්චාවචන ගුණකය", "ஏற்ற இறக்க பெருக்கி"),
    "Initial return shock (%)": ("ආරම්භක ප්‍රතිලාභ කම්පනය (%)", "ஆரம்ப வருவாய் அதிர்ச்சி (%)"),
    "2022 crisis scenario": ("2022 අර්බුද අවස්ථාව", "2022 நெருக்கடி சூழ்நிலை"),
    "Reset": ("යළි සකසන්න", "மீட்டமை"),
    "This company": ("මෙම සමාගම", "இந்த நிறுவனம்"),
    "All companies": ("සියලු සමාගම්", "அனைத்து நிறுவனங்கள்"),
    "Why this prediction? See which signals drove it": ("මෙම පුරෝකථනය ඇයි? බලපෑ සංඥා බලන්න",
                                                         "ஏன் இந்த முன்னறிவிப்பு? காரணமான சமிக்ஞைகளைப் பாருங்கள்"),
    "Research prototype for decision support only. Not financial advice. Predictions are probabilistic, may be "
    "wrong, and carry no guarantee of returns.": (
        "තීරණ ගැනීමට සහාය වීම සඳහා පමණක් වූ පර්යේෂණ මූලාකෘතියකි. මෙය මූල්‍ය උපදෙසක් නොවේ. පුරෝකථන සම්භාවිතාමය "
        "වන අතර වැරදි විය හැකි අතර ප්‍රතිලාභ සඳහා කිසිදු සහතිකයක් නැත.",
        "முடிவெடுப்பதற்கு உதவுவதற்கான ஆய்வு முன்மாதிரி மட்டுமே. இது நிதி ஆலோசனை அல்ல. முன்னறிவிப்புகள் நிகழ்தகவு "
        "அடிப்படையிலானவை, தவறாக இருக்கலாம், வருவாய்க்கு எந்த உத்தரவாதமும் இல்லை."),
}

GUIDES: dict[str, dict[str, list[str]]] = {
    "Market overview": {
        "en": ["Choose a stock and a date in the left panel; every page uses that choice.",
               "This page shows the whole market: index levels, today's biggest movers and the news mood.",
               "Go to Stock prediction to see what the model expects for your chosen stock."],
        "si": ["වම් පැත්තේ කොටසක් සහ දිනයක් තෝරන්න; සෑම පිටුවක්ම එම තේරීම භාවිත කරයි.",
               "මෙම පිටුව සමස්ත වෙළඳපොළ පෙන්වයි: දර්ශක, වැඩිපුරම වෙනස් වූ කොටස් සහ පුවත් මනෝභාවය.",
               "ඔබ තෝරාගත් කොටස ගැන ආකෘතිය අපේක්ෂා කරන දේ දැකීමට 'කොටස් පුරෝකථනය' වෙත යන්න."],
        "ta": ["இடது பக்கத்தில் ஒரு பங்கையும் திகதியையும் தேர்வு செய்யுங்கள்; ஒவ்வொரு பக்கமும் அதைப் பயன்படுத்தும்.",
               "இந்தப் பக்கம் முழு சந்தையைக் காட்டுகிறது: சுட்டெண்கள், அதிகம் மாறிய பங்குகள், செய்தி உணர்வு.",
               "நீங்கள் தேர்ந்த பங்கிற்கு மாதிரி எதிர்பார்ப்பதைப் பார்க்க 'பங்கு முன்னறிவிப்பு' பக்கத்துக்குச் செல்லுங்கள்."],
    },
    "Stock prediction": {
        "en": ["The cards show the model's call for the next day and the next five days.",
               "'Unclear' means the chance of a rise is too close to call; 'strong signal' calls were right more often in past tests.",
               "Use it as one input alongside your own judgement, never as the only reason to trade."],
        "si": ["කාඩ්පත් මගින් ඊළඟ දිනය සහ ඊළඟ දින පහ සඳහා ආකෘතියේ පුරෝකථනය පෙන්වයි.",
               "'අපැහැදිලියි' යනු ඉහළ යාමේ සම්භාවිතාව තීරණය කිරීමට තරම් පැහැදිලි නොවන බවයි; 'ශක්තිමත් සංඥා' අතීත පරීක්ෂාවලදී බොහෝ විට නිවැරදි විය.",
               "මෙය ඔබේම විනිශ්චය සමඟ එක් ආදානයක් ලෙස පමණක් භාවිත කරන්න."],
        "ta": ["அட்டைகள் அடுத்த நாள் மற்றும் அடுத்த ஐந்து நாட்களுக்கான மாதிரியின் கணிப்பைக் காட்டுகின்றன.",
               "'தெளிவில்லை' என்றால் உயர்வுக்கான வாய்ப்பு தீர்மானிக்க முடியாத அளவு நெருக்கமானது; 'வலுவான சமிக்ஞை' கணிப்புகள் கடந்த சோதனைகளில் அடிக்கடி சரியாக இருந்தன.",
               "இதை உங்கள் சொந்த தீர்ப்புடன் ஒரு உள்ளீடாக மட்டும் பயன்படுத்துங்கள்."],
    },
    "Explainability": {
        "en": ["Green bars pushed the prediction towards 'up', red bars towards 'down'.",
               "Longer bars mattered more. The plain-language box summarises the same thing in words.",
               "Explanations show how the model reasoned, not that it is right."],
        "si": ["කොළ තීරු පුරෝකථනය 'ඉහළට' දෙසට ද, රතු තීරු 'පහළට' දෙසට ද තල්ලු කළේය.",
               "දිගු තීරු වඩා වැදගත් විය. සරල භාෂා කොටුව එයම වචනවලින් සාරාංශ කරයි.",
               "පැහැදිලි කිරීම් පෙන්වන්නේ ආකෘතිය සිතූ ආකාරය මිස එය නිවැරදි බව නොවේ."],
        "ta": ["பச்சை பட்டைகள் கணிப்பை 'உயர்வு' நோக்கியும், சிவப்பு பட்டைகள் 'இறக்கம்' நோக்கியும் தள்ளின.",
               "நீளமான பட்டைகள் அதிக முக்கியத்துவம் பெற்றன. எளிய மொழிப் பெட்டி அதையே சொற்களில் சுருக்குகிறது.",
               "விளக்கங்கள் மாதிரி எப்படி சிந்தித்தது என்பதைக் காட்டுகின்றன, அது சரி என்பதை அல்ல."],
    },
    "Risk & stress test": {
        "en": ["Value at risk is the loss you should exceed only rarely (5% or 1% of days) for the position size you enter.",
               "Move the sliders or press '2022 crisis scenario' to see losses in a bad market.",
               "Expert details at the bottom show how well these estimates matched history."],
        "si": ["අවදානම් අගය (VaR) යනු ඔබ ඇතුළත් කරන ආයෝජනයට කලාතුරකින් පමණක් (දින 5% හෝ 1%) ඉක්මවිය යුතු පාඩුවයි.",
               "නරක වෙළඳපොළක පාඩු දැකීමට ස්ලයිඩර් වෙනස් කරන්න හෝ '2022 අර්බුද අවස්ථාව' ඔබන්න.",
               "පහළ ඇති විශේෂඥ විස්තර මගින් මෙම ඇස්තමේන්තු ඉතිහාසයට කෙතරම් ගැළපුණාද යන්න පෙන්වයි."],
        "ta": ["இடர் மதிப்பு (VaR) என்பது நீங்கள் உள்ளிடும் முதலீட்டுக்கு அரிதாக மட்டுமே (5% அல்லது 1% நாட்கள்) மீறப்பட வேண்டிய இழப்பு.",
               "மோசமான சந்தையில் இழப்புகளைப் பார்க்க ஸ்லைடர்களை நகர்த்துங்கள் அல்லது '2022 நெருக்கடி சூழ்நிலை' அழுத்துங்கள்.",
               "கீழே உள்ள நிபுணர் விவரங்கள் இந்த மதிப்பீடுகள் வரலாற்றுடன் எவ்வளவு பொருந்தின என்பதைக் காட்டுகின்றன."],
    },
    "News sentiment": {
        "en": ["The line shows the overall mood of business news in English, Sinhala and Tamil (−1 negative, +1 positive).",
               "Dots mark days with news about your chosen company.",
               "The list below gives headlines and links only; open the article to read it in full."],
        "si": ["රේඛාව ඉංග්‍රීසි, සිංහල සහ දෙමළ ව්‍යාපාරික පුවත්වල සමස්ත මනෝභාවය පෙන්වයි (−1 සෘණ, +1 ධන).",
               "තිත් මගින් ඔබ තෝරාගත් සමාගම ගැන පුවත් ඇති දින සලකුණු කරයි.",
               "පහත ලැයිස්තුවේ ඇත්තේ සිරස්තල සහ සබැඳි පමණි; සම්පූර්ණ ලිපිය කියවීමට එය විවෘත කරන්න."],
        "ta": ["கோடு ஆங்கிலம், சிங்களம், தமிழ் வணிகச் செய்திகளின் ஒட்டுமொத்த மனநிலையைக் காட்டுகிறது (−1 எதிர்மறை, +1 நேர்மறை).",
               "புள்ளிகள் நீங்கள் தேர்ந்த நிறுவனம் பற்றிய செய்தி உள்ள நாட்களைக் குறிக்கின்றன.",
               "கீழே உள்ள பட்டியலில் தலைப்புகளும் இணைப்புகளும் மட்டுமே; முழுக் கட்டுரையைப் படிக்க அதைத் திறக்கவும்."],
    },
}


def lang() -> str:
    return st.session_state.get("lang", "en")


def tr(text: str) -> str:
    code = lang()
    if code == "en" or text not in T:
        return text
    return T[text][0 if code == "si" else 1]


def guide(page: str) -> None:
    lines = GUIDES.get(page, {}).get(lang()) or GUIDES.get(page, {}).get("en")
    if lines:
        with st.expander(tr("How to use this page"), expanded=True):
            st.markdown("\n".join(f"- {l}" for l in lines))
