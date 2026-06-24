"""Demographic survey analysis.

Generates ~60 charts covering age, education, religious observance, language
proficiency, occupation, and birth origin. Outputs to charts/us or charts/global.

Usage:
    python3 analysis.py          # US-only (default)
    python3 analysis.py us       # US-only
    python3 analysis.py global   # All respondents
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter
from pathlib import Path

# ── Configuration ───────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 13,
    "axes.titlesize": 16, "axes.labelsize": 14, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "legend.fontsize": 11, "figure.facecolor": "white",
})

LINE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#000075", "#dcbeff",
]
PALETTE = plt.cm.tab20.colors

SCOPE = sys.argv[1] if len(sys.argv) > 1 else "us"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "charts" / SCOPE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AGE_ORDER = ["Under 20", "20-29", "30-39", "40-49", "50-59", "60+"]
OBS_ORDER = ["Non-observant", "Traditional", "Observant", "Very Observant"]
FARSI_ORDER = [
    "Don't know it at all", "Only know food names and swear words",
    "Limited Proficiency", "Fluent",
]
EDU_ORDER = [
    "Some High School", "High School diploma or GED", "Associates Degree",
    "Bachelors Degree", "Masters Degree", "Doctorate or PhD",
]
EDU_SHORT = dict(zip(EDU_ORDER, ["<HS", "HS/GED", "Associates", "Bachelors", "Masters", "Doctorate"]))

HEALTHCARE_CATS = [
    "Physician", "Dentist / Dental", "Pharmacist", "Nurse / NP",
    "Physician Assistant", "Occupational Therapist", "Physical Therapist",
    "Speech Therapist", "Optometrist / Optician", "Other Allied Health",
]
OCCUPATION_ORDER = [
    *HEALTHCARE_CATS,
    "Law / Legal", "Accounting / CPA", "Finance / Banking", "Real Estate",
    "Tech / IT", "Engineering", "Education", "Business Owner", "Management",
    "Sales", "Barber / Beauty", "Jewelry", "Tailor / Cobbler",
    "Construction / Trades", "Driver / Transport", "Food / Hospitality",
    "Arts / Media / Marketing", "Social Work / Community",
    "Student", "Retired", "Homemaker", "Other",
]
REGION_ORDER = ["Central Asia / FSU", "Israel", "New York City", "Other US", "Europe / Other"]

FARSI_NUMERIC = {"Don't know it at all": 1, "Only know food names and swear words": 2,
                 "Limited Proficiency": 3, "Fluent": 4}
EDU_NUMERIC = dict(zip(EDU_ORDER, range(1, 7)))
OBS_NUMERIC = dict(zip(OBS_ORDER, range(1, 5)))

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════════════

df = pd.read_csv(BASE_DIR / "Raw_Census_2021.csv", low_memory=False)
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
df.columns = df.columns.str.strip()

# ── Multilingual consolidation ──────────────────────────────────────────

df["observance"] = df["observance"].replace({
    "Традиционный": "Traditional", "Строго соблюдающий": "Very Observant",
    "Не соблюдающий": "Non-observant", "דתי": "Observant",
    "מסורתי": "Traditional", "לא שומר תורה ומצוות": "Non-observant",
    "No Answer Provided": np.nan,
})

df["farsi"] = df["farsi"].replace({
    "с небольшими ограничениями": "Limited Proficiency",
    "В совершенстве": "Fluent",
    "понимаю только названия еды и нецензурные слова": "Only know food names and swear words",
    "не знаю вообще": "Don't know it at all",
    "בינונית": "Limited Proficiency",
    "אני מבין רק את שמות האוכל והמילים המגונות": "Only know food names and swear words",
    "מצויין": "Fluent", "בכלל לא יודע": "Don't know it at all",
    "No Answer Provided": np.nan,
})

df["education"] = df["education"].replace({
    "окончил(а) институт": "Bachelors Degree", "техникум": "Associates Degree",
    "среднее": "High School diploma or GED", "кандидат наук": "Masters Degree",
    "доктор наук": "Doctorate or PhD", "неполное среднее": "Some High School",
    "תואר ראשון": "Bachelors Degree", "מכללה טכנית": "Associates Degree",
    "תואר שני": "Masters Degree", "בית ספר תיכון": "High School diploma or GED",
    "Doctorate or PhD": "Doctorate or PhD", "No Answer Provided": np.nan,
})

df["age"] = df["age"].replace({"60 +": "60+"})

df["is_member"] = df["is_member"].replace(
    {"Да": "Yes", "כן": "Yes", "Нет": "No", "לא": "No"}
)

df["current_loc"] = df["current_loc"].replace({
    "США": "United States", "ארה&quot;ב": "United States",
    "Израиль": "Israel", "ישראל": "Israel", "В Израиле": "Israel",
    "Канада": "Canada", "Средняя Азиа": "Central Asia",
    "Central Asia": "Central Asia", "Евросоюз": "European Union",
    "Россия/Украина": "Russia/Ukraine", "Другое": "Other",
    "No Answer Provided": np.nan,
})

# ── Birth city normalization ────────────────────────────────────────────

_CITY_LOOKUP = {
    "tashkent": "Tashkent", "ташкент": "Tashkent", "ташкентх": "Tashkent",
    "ташкеит": "Tashkent", "ташкенте": "Tashkent", "taskent": "Tashkent",
    "tashekent": "Tashkent", "taschkent": "Tashkent", "toshkent": "Tashkent",
    "טשכנט": "Tashkent",
    "samarkand": "Samarkand", "самарканд": "Samarkand", "camarkand": "Samarkand",
    "smarkand": "Samarkand", "sanarkand": "Samarkand", "samarqand": "Samarkand",
    "סמרקנד": "Samarkand", "סמאקנד": "Samarkand",
    "samarkand uzbekistan": "Samarkand", "samarkand ussr": "Samarkand",
    "samarkand uz": "Samarkand", "uzbekistan samarkand": "Samarkand",
    "uzbekistan samarkakd": "Samarkand",
    "bukhara": "Bukhara", "бухара": "Bukhara", "бухаре": "Bukhara",
    "buhara": "Bukhara", "buchara": "Bukhara", "bychara": "Bukhara",
    "byshara": "Bukhara", "buharara": "Bukhara", "בוכרה": "Bukhara",
    "bukhara uzbekistan": "Bukhara",
    "dushanbe": "Dushanbe", "душанбе": "Dushanbe", "דושנבה": "Dushanbe",
    "dushambe": "Dushanbe", "dushambeh": "Dushanbe", "duschanbe": "Dushanbe",
    "paishanbe": "Dushanbe", "paishambe": "Dushanbe",
    "dushanbe tadjikistan": "Dushanbe", "dushanbe tadzhikistan": "Dushanbe",
    "tadjistan /dushanbe": "Dushanbe",
    "fergana": "Fergana", "фергана": "Fergana", "ferghana": "Fergana",
    "fergana uzbekistan": "Fergana", "узбекистан фергана": "Fergana",
    "andizhan": "Andijan", "andijan": "Andijan", "andizan": "Andijan",
    "andijon": "Andijan", "андижан": "Andijan", "undizan": "Andijan",
    "andezhan": "Andijan", "andijhan uzbekistan": "Andijan",
    "andizhanuzbekistan": "Andijan",
    "kokand": "Kokand", "коканд": "Kokand", "коканд узбекистан": "Kokand",
    "kokand uzbekistan": "Kokand",
    "navoi": "Navoi", "навои": "Navoi", "navaii": "Navoi", "navai": "Navoi",
    "navoii": "Navoi", "novoi": "Navoi", "navio": "Navoi",
    "kattakurgan": "Kattakurgan", "katakurgan": "Kattakurgan",
    "kattaqurghon": "Kattakurgan", "катакурган": "Kattakurgan",
    "katta kurgan": "Kattakurgan",
    "margilan": "Margilan", "margelan": "Margilan", "маргелан": "Margilan",
    "маргилан": "Margilan", "margilan uzbekistan": "Margilan",
    "chimkent": "Chimkent", "чимкент": "Chimkent", "chenkent": "Chimkent",
    "shymkent": "Chimkent", "chimkent kazakstan": "Chimkent",
    "chimkent kazakhstan": "Chimkent",
    "leninabad": "Leninabad", "ленинабад": "Leninabad", "leninbad": "Leninabad",
    "linlabad": "Leninabad", "leninobad": "Leninabad",
    "leninabad tashkent": "Leninabad", "khujand": "Leninabad",
    "khuzhand": "Leninabad", "hodgent": "Leninabad",
    "namangan": "Namangan", "наманган": "Namangan", "nsmangan": "Namangan",
    "denau": "Denau", "денау": "Denau",
    "shahrisabz": "Shahrisabz", "шахрисабз": "Shahrisabz",
    "shakrisabz": "Shahrisabz", "shaahrisabz": "Shahrisabz",
    "shachrisabs": "Shahrisabz", "shahrisabs": "Shahrisabz",
    "tel aviv": "Tel Aviv", "telaviv": "Tel Aviv", "tel aviv israel": "Tel Aviv",
    "moscow": "Moscow", "мoscow": "Moscow",
    "israel": "Israel", "isreal": "Israel", "ישראל": "Israel",
}

_NYC_KEYS = {
    "new york", "new york city", "nyc", "ny", "new york ny",
    "new york queens", "usa  brooklyn ny", "manhattan ny", "manhattan", "kew gardens",
}
_NYC_NEIGHBORHOODS = {
    "queens": "Queens, NYC", "queens ny": "Queens, NYC",
    "queens new york city": "Queens, NYC", "bayside queens": "Queens, NYC",
    "forest hills": "Forest Hills, NYC", "forest hills ny": "Forest Hills, NYC",
    "forrest hills": "Forest Hills, NYC", "forest hulls": "Forest Hills, NYC",
    "flushing": "Flushing, NYC", "flushing ny": "Flushing, NYC",
    "rego park": "Rego Park, NYC", "brooklyn": "Brooklyn, NYC",
    "corona": "Corona, NYC", "fresh meadows": "Fresh Meadows, NYC",
}


def _normalize_birth_city(val):
    if pd.isna(val):
        return val
    s = str(val).strip()
    low = s.lower().replace(",", "").replace(".", "").replace("-", "")
    if low in _CITY_LOOKUP:
        return _CITY_LOOKUP[low]
    if low in _NYC_KEYS:
        return "New York City"
    if low in _NYC_NEIGHBORHOODS:
        return _NYC_NEIGHBORHOODS[low]
    return s


df["birth_city"] = df["birth_city"].apply(_normalize_birth_city)

# ── Derived columns ────────────────────────────────────────────────────

_CA_CITIES = {
    "Tashkent", "Samarkand", "Bukhara", "Dushanbe", "Fergana", "Andijan",
    "Kokand", "Navoi", "Kattakurgan", "Margilan", "Chimkent", "Leninabad",
    "Namangan", "Denau", "Shahrisabz", "Moscow", "Tajikistan", "Uzbekistan",
    "Kazakhstan", "Tadzhikistan", "Turkestan", "Turkmenistan", "Bishkek", "Frunze",
}
_IL_CITIES = {"Tel Aviv", "Israel", "Jerusalem", "Holon", "Ramla", "Ashdod"}
_NYC_TAGS = ["NYC", "Queens", "Brooklyn", "Flushing", "Forest Hills",
             "Fresh Meadows", "Rego Park", "Corona", "New York"]
_OTHER_US = {"Long Island", "San Diego", "Cleveland", "Miami", "Los Angeles",
             "United States", "Cincinnati", "New Haven"}
_EURO = {"Vienna", "Berlin", "Hannover", "Düsseldorf", "Odessa", "Kiev", "Toronto"}


def _birth_region(city):
    if pd.isna(city):
        return np.nan
    if city in _CA_CITIES:
        return "Central Asia / FSU"
    if city in _IL_CITIES or "israel" in str(city).lower():
        return "Israel"
    if any(tag in str(city) for tag in _NYC_TAGS):
        return "New York City"
    if city in _OTHER_US or "US" in str(city):
        return "Other US"
    if city in _EURO or "Canada" in str(city):
        return "Europe / Other"
    return "Central Asia / FSU"


df["birth_region"] = df["birth_city"].apply(_birth_region)
df["num_home_langs"] = df["home_lang"].apply(
    lambda v: len([l for l in str(v).split(".") if l.strip()]) if pd.notna(v) else np.nan
)

# ── Occupation categorization ───────────────────────────────────────────

_OCC_RULES = [
    ("Physician", ["physician", "doctor", "md ", "surgeon", "врач", "pediatric",
                   "internist", "cardiolog", "urolog", "anesthesi", "radiolog",
                   "psychiatr", "neurolog", "oncolog", "dermatolog", "medical director"]),
    ("Dentist / Dental", ["dentist", "dental", "orthodon", "endodont", "periodon",
                          "oral surg", "стоматолог"]),
    ("Pharmacist", ["pharmacist", "pharmacy", "фармацевт"]),
    ("Optometrist / Optician", ["optometr", "optician", "eye doctor"]),
    ("Nurse / NP", ["nurse", "rn", "nursing", "медсестра", "медбрат"]),
    ("Occupational Therapist", ["occupational therap", "ot ", "cota"]),
    ("Physical Therapist", ["physical therap", "pt ", "physiotherap"]),
    ("Speech Therapist", ["speech", "slp", "speech pathol", "логопед"]),
    ("Other Allied Health", ["respiratory", "hemodialysis", "medical assist",
                             "medical biller", "medical coder", "hha", "home health",
                             "medical tech", "phlebotom", "x-ray", "sonograph",
                             "ultrasound", "behavior analyst", "bcba"]),
    ("Student", ["student", "school", "studying", "in college", "учусь", "студент", "учеба"]),
    ("Law / Legal", ["lawyer", "attorney", "paralegal", "legal", "адвокат", "юрист"]),
    ("Accounting / CPA", ["accountant", "accounting", "cpa", "bookkeep", "auditor",
                          "бухгалтер", "tax preparer"]),
    ("Finance / Banking", ["finance", "banking", "financial", "investment", "mortgage",
                           "loan", "credit"]),
    ("Real Estate", ["real estate", "realtor", "realty", "broker"]),
    ("Tech / IT", ["software", "developer", "programmer", "coding", "it ", "cyber",
                   "data", "web ", "программист", "computer", "devops", "qa",
                   "информатик", "tech lead", "full stack"]),
    ("Engineering", ["engineer", "engineering", "инженер", "архитект", "architect"]),
    ("Education", ["teacher", "professor", "education", "tutor", "educator", "pedagog",
                   "педагог", "преподав", "учитель", "instructor", "special ed"]),
    ("Business Owner", ["business", "entrepreneur", "ceo", "owner", "self employ",
                        "self-employ", "предприним", "бизнес"]),
    ("Management", ["manager", "director", "supervisor", "менеджер", "управля",
                    "coordinator", "администрат"]),
    ("Sales", ["sales", "продавец", "продажи"]),
    ("Barber / Beauty", ["barber", "парикмахер", "hairdress", "hairstyl", "cosmetol",
                         "beauty", "nail tech", "estheti", "makeup"]),
    ("Jewelry", ["jewel", "ювелир"]),
    ("Tailor / Cobbler", ["cobbler", "tailor", "портной", "сапожник", "shoemaker",
                          "seamstress", "швея"]),
    ("Construction / Trades", ["contractor", "construction", "electrician", "plumb",
                               "hvac", "строител", "mechanic", "welding", "carpent"]),
    ("Driver / Transport", ["driver", "водитель", "trucker", "cdl", "taxi", "uber", "delivery"]),
    ("Retired", ["retire", "пенсионер", "пенсионерка"]),
    ("Homemaker", ["house wife", "housewife", "homemaker", "stay at home",
                   "домохозяйка", "домохоз"]),
    ("Food / Hospitality", ["cook", "chef", "restaurant", "повар", "baker", "кулинар",
                            "catering", "food"]),
    ("Arts / Media / Marketing", ["photograph", "musician", "музыкант", "artist", "дизайн",
                                  "design", "graphic", "film", "video", "media",
                                  "marketing", "журнали", "writer", "journal"]),
    ("Social Work / Community", ["social work", "rabbi", "раввин", "clergy", "nonprofit",
                                 "community", "government", "counselor", "психолог", "therapist"]),
]
_OCC_EXACT = {"pa": "Physician Assistant", "ot": "Occupational Therapist", "pt": "Physical Therapist"}
_OCC_KEYWORDS_PA = ["physician assistant", " pa ", "pa-c"]
_OCC_EMPTY = {"-", "n/a", "no answer provided", "", "none", "-none-"}


def _categorize_vocation(val):
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    if v in _OCC_EMPTY:
        return np.nan
    if v in _OCC_EXACT:
        return _OCC_EXACT[v]
    if any(k in v for k in _OCC_KEYWORDS_PA):
        return "Physician Assistant"
    for category, keywords in _OCC_RULES:
        if category == "Engineering" and "software" in v:
            continue
        if any(k in v for k in keywords):
            return category
    return "Other"


df["occupation"] = df["vocation"].apply(_categorize_vocation)

# ── Numeric scores ──────────────────────────────────────────────────────

df["farsi_num"] = df["farsi"].map(FARSI_NUMERIC)
df["edu_num"] = df["education"].map(EDU_NUMERIC)
df["obs_num"] = df["observance"].map(OBS_NUMERIC)
df["is_healthcare"] = df["occupation"].isin(HEALTHCARE_CATS)
df["born_us"] = df["birth_region"].isin(["New York City", "Other US"])

# ── Scope filter ────────────────────────────────────────────────────────

if SCOPE == "us":
    df = df[df["country"] == "US"].copy()
    scope_label = "United States"
else:
    scope_label = "Global"

print(f"Scope: {scope_label} | Respondents: {len(df)}")

# ═══════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _save(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png")


def _reindex(ct, rows=None, cols=None):
    if rows:
        ct = ct.reindex(index=[r for r in rows if r in ct.index])
    if cols:
        ct = ct.reindex(columns=[c for c in cols if c in ct.columns])
    return ct.dropna(how="all", axis=0).dropna(how="all", axis=1)


def bar_chart(series, title, xlabel, name, ylabel="Number of Respondents",
              horizontal=False, order=None, figsize=(10, 6)):
    counts = series.dropna().value_counts()
    if order:
        counts = counts.reindex([o for o in order if o in counts.index]).dropna()
    total = counts.sum()
    fig, ax = plt.subplots(figsize=figsize)
    colors = list(PALETTE[: len(counts)])
    if horizontal:
        counts.plot.barh(ax=ax, color=colors)
        ax.set_xlabel(ylabel); ax.set_ylabel(xlabel); ax.invert_yaxis()
        for i, v in enumerate(counts.values):
            ax.text(v + counts.max() * 0.01, i,
                    f" {v} ({v/total*100:.1f}%)", va="center", fontsize=11)
    else:
        counts.plot.bar(ax=ax, color=colors)
        ax.set_ylabel(ylabel); ax.set_xlabel(xlabel)
        for i, v in enumerate(counts.values):
            ax.text(i, v + counts.max() * 0.01,
                    f"{v}\n({v/total*100:.1f}%)", ha="center", fontsize=10)
    ax.set_title(f"{title} ({scope_label})")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    _save(fig, name)


def pie_chart(series, title, name, figsize=(8, 8)):
    counts = series.dropna().value_counts()
    total = counts.sum()
    labels = [f"{lbl}\n(n={v})" for lbl, v in zip(counts.index, counts.values)]
    fig, ax = plt.subplots(figsize=figsize)
    autopct = lambda pct: f"{pct:.1f}%" if pct >= 3 else ""
    _, texts, autotexts = ax.pie(
        counts.values, labels=labels, autopct=autopct,
        colors=PALETTE[: len(counts)], startangle=140, pctdistance=0.8)
    for t in [*texts, *autotexts]:
        t.set_fontsize(12)
    ax.set_title(f"{title} ({scope_label})  [N={total}]")
    plt.tight_layout()
    _save(fig, name)


def hbar_with_pct(series, title, name, n=20, figsize=(10, 7)):
    counts = series.dropna().value_counts().head(n)
    total = series.dropna().shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    counts.plot.barh(ax=ax, color=PALETTE[: len(counts)])
    ax.set_title(f"{title} ({scope_label})  [N={total}]")
    ax.set_xlabel("Number of Respondents"); ax.invert_yaxis()
    for i, v in enumerate(counts.values):
        ax.text(v + counts.max() * 0.01, i,
                f" {v} ({v/total*100:.1f}%)", va="center", fontsize=11)
    plt.tight_layout()
    _save(fig, name)


def stacked_bar(ct, title, name, cmap="Set2", ylabel="Count",
                pct=False, legend_title="", figsize=(11, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ct.plot.bar(ax=ax, stacked=True, colormap=cmap)
    ax.set_title(f"{title} ({scope_label})")
    ax.set_ylabel(ylabel)
    if pct:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    _save(fig, name)


def stacked_barh(ct, title, name, cmap="Set2", legend_title="", figsize=(13, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_norm.plot.barh(ax=ax, stacked=True, colormap=cmap)
    ax.set_title(f"{title} ({scope_label})")
    ax.set_xlabel("Percentage"); ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.invert_yaxis(); plt.tight_layout()
    _save(fig, name)


def line_chart(ct_norm, title, name, category_order, label_map=None,
               ylabel="% of Group", xlabel="", figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    for idx, cat in enumerate(category_order):
        if cat in ct_norm.columns:
            label = (label_map or {}).get(cat, cat)
            ax.plot(ct_norm.index, ct_norm[cat].values, "o-", label=label,
                    color=LINE_COLORS[idx % len(LINE_COLORS)], linewidth=2.5, markersize=8)
    ax.set_title(f"{title} ({scope_label})")
    ax.set_ylabel(ylabel); ax.set_xlabel(xlabel)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(True, alpha=0.3); ax.legend()
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    _save(fig, name)


def avg_score_lines(data, group_col, score_col, line_col, line_order,
                    title, name, ylabel, group_order=None,
                    ylim=None, yticks=None, yticklabels=None, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    sub = data[data[group_col].isin(group_order or []) & data[score_col].notna()].copy()
    for idx, lval in enumerate(line_order):
        s = sub[sub[line_col] == lval] if line_col != "_all" else sub
        means = s.groupby(group_col)[score_col].mean()
        if group_order:
            means = means.reindex(group_order).dropna()
        ax.plot(means.index, means.values, "o-", label=f"{lval} (n={len(s)})",
                color=LINE_COLORS[idx % len(LINE_COLORS)], linewidth=2.5, markersize=8)
    ax.set_title(f"{title} ({scope_label})")
    ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)
    if yticks:
        ax.set_yticks(yticks)
    if yticklabels:
        ax.set_yticklabels(yticklabels)
    ax.legend(title=line_col.replace("_", " ").title() if line_col != "_all" else "")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    _save(fig, name)


# Cross-tab convenience: bar + optional normalized + optional line
def crosstab_suite(row_col, col_col, row_order, col_order, base_name, base_title,
                   cmap="Set2", legend_title="", label_map=None,
                   do_raw=True, do_pct=True, do_line=True):
    ct = _reindex(pd.crosstab(df[row_col], df[col_col]), row_order, col_order)
    ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
    if do_raw:
        stacked_bar(ct, base_title, f"{base_name}", cmap=cmap, legend_title=legend_title)
    if do_pct:
        stacked_bar(ct_norm, f"{base_title} — Normalized", f"{base_name}_pct",
                    cmap=cmap, ylabel="Percentage", pct=True, legend_title=legend_title)
    if do_line:
        line_chart(ct_norm, base_title.replace("by", "(%) by"),
                   f"{base_name.rstrip('_')}b_lines", col_order, label_map=label_map)
    return ct, ct_norm


# ═══════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════

ages_present = [a for a in AGE_ORDER if a in df["age"].values]

# ── 01–04: Basic distributions ──────────────────────────────────────────

print("01–04: Basic distributions")
bar_chart(df["age"], "Age Distribution", "Age Group", "01_age_distribution", order=AGE_ORDER)
pie_chart(df["age"], "Age Distribution", "01_age_pie")
bar_chart(df["education"], "Education Level", "Education", "02_education_bar", order=EDU_ORDER)
pie_chart(df["education"], "Education Level", "02_education_pie")
bar_chart(df["observance"], "Religious Observance", "Observance", "03_observance_bar", order=OBS_ORDER)
pie_chart(df["observance"], "Religious Observance", "03_observance_pie")
bar_chart(df["farsi"], "Farsi Proficiency", "Level", "04_farsi_proficiency",
          order=FARSI_ORDER, horizontal=True, figsize=(10, 5))

# ── 05: Home languages ─────────────────────────────────────────────────

print("05: Home languages")
_LANG_MAP = {
    "русский": "Russian", "иврит": "Hebrew", "עִברִית": "Hebrew",
    "английский": "English", "бухарский": "Bukharian", "רוסית": "Russian",
    "עברית": "Hebrew", "אנגלית": "English", "בוכרית": "Bukharian",
    "תורכי": "Other", "другой": "Other",
}
lang_counter = Counter()
for langs in df["home_lang"].dropna():
    for lang in str(langs).split("."):
        lang = lang.strip()
        if lang:
            lang_counter[_LANG_MAP.get(lang, lang)] += 1
lang_s = pd.Series(lang_counter).sort_values(ascending=False)
N = len(df)
fig, ax = plt.subplots(figsize=(10, 5))
lang_s.plot.bar(ax=ax, color=PALETTE[: len(lang_s)])
ax.set_title(f"Languages Spoken at Home ({scope_label})  [N={N}, multi-select]")
ax.set_ylabel("Respondents"); ax.set_xlabel("Language")
for i, v in enumerate(lang_s.values):
    ax.text(i, v + lang_s.max() * 0.01, f"{v}\n({v/N*100:.0f}%)", ha="center", fontsize=10)
plt.xticks(rotation=45, ha="right"); plt.tight_layout()
_save(fig, "05_home_languages")

# ── 06: Location ───────────────────────────────────────────────────────

print("06: Location")
if SCOPE == "us":
    bar_chart(df["region"].dropna(), "US State Distribution", "State",
              "06_us_state", horizontal=True, figsize=(10, 8))
else:
    bar_chart(df["current_loc"].dropna(), "Country of Residence", "Country",
              "06_current_location", horizontal=True)

# ── 07–08: Birth city & membership ─────────────────────────────────────

print("07–08: Birth city & membership")
hbar_with_pct(df["birth_city"], "Birth City — Top 20", "07_birth_city")
pie_chart(df["is_member"], "Community Membership", "08_membership")

# ── 09–13: Cross-tabulations ───────────────────────────────────────────

print("09–13: Cross-tabulations")
crosstab_suite("age", "observance", AGE_ORDER, OBS_ORDER,
               "09_age_x_observance", "Observance by Age Group", legend_title="Observance")
crosstab_suite("age", "education", AGE_ORDER, EDU_ORDER,
               "10_age_x_education", "Education by Age Group",
               cmap="tab20", legend_title="Education", label_map=EDU_SHORT, do_pct=False)
crosstab_suite("age", "farsi", AGE_ORDER, FARSI_ORDER,
               "11_age_x_farsi", "Farsi by Age Group",
               cmap="RdYlGn", legend_title="Farsi Level", do_raw=False)
crosstab_suite("observance", "education", OBS_ORDER, EDU_ORDER,
               "12_observance_x_education", "Education by Observance",
               cmap="tab20", legend_title="Education", label_map=EDU_SHORT, do_raw=False)
crosstab_suite("observance", "farsi", OBS_ORDER, FARSI_ORDER,
               "13_observance_x_farsi", "Farsi by Observance",
               cmap="RdYlGn", legend_title="Farsi Level", do_raw=False)

# ── 14: Heatmap ────────────────────────────────────────────────────────

print("14: Heatmap")
heat = _reindex(pd.crosstab(df["farsi"], df["age"]), FARSI_ORDER, AGE_ORDER)
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(heat.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(heat.columns))); ax.set_xticklabels(heat.columns, rotation=45, ha="right")
ax.set_yticks(range(len(heat.index))); ax.set_yticklabels(heat.index)
for i in range(len(heat.index)):
    for j in range(len(heat.columns)):
        v = heat.values[i, j]
        ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=12,
                color="white" if v > heat.values.max() * 0.6 else "black")
fig.colorbar(im, ax=ax, label="Count")
ax.set_title(f"Farsi × Age ({scope_label})")
plt.tight_layout(); _save(fig, "14_farsi_age_heatmap")

# ── 15–16: Faceted cross-tabs by age + line summaries ──────────────────

print("15–16: Faceted cross-tabs")
for conf in [
    ("farsi", FARSI_ORDER, "RdYlGn", "Farsi Proficiency", "15", "farsi_num",
     "Avg. Farsi by Observance × Age", (0.8, 4.2), "1=None → 4=Fluent"),
    ("education", EDU_ORDER, "tab20", "Education", "16", "edu_num",
     "Avg. Education by Observance × Age", (0.5, 6.5), "1=<HS → 6=Doctorate"),
]:
    col, col_order, cmap, legend, num, score_col, line_title, ylim, ylab = conf
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes_flat = axes.flatten()
    all_handles = {}
    display_order = [EDU_SHORT.get(c, c) for c in col_order] if col == "education" else col_order
    for idx, age in enumerate(ages_present):
        ax = axes_flat[idx]
        sub = df[df["age"] == age]
        ct = _reindex(pd.crosstab(sub["observance"], sub[col]), OBS_ORDER, col_order)
        if ct.empty:
            ax.set_visible(False); continue
        if col == "education":
            ct.columns = [EDU_SHORT.get(c, c) for c in ct.columns]
        ct_n = ct.div(ct.sum(axis=1), axis=0) * 100
        ct_n.plot.bar(ax=ax, stacked=True, colormap=cmap, legend=False)
        ax.set_title(f"Age {age} (n={len(sub)})", fontsize=12)
        ax.set_ylabel("%" if idx % 3 == 0 else ""); ax.set_xlabel("")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.tick_params(axis="x", rotation=45)
        for h, l in zip(*ax.get_legend_handles_labels()):
            all_handles[l] = h
    for idx in range(len(ages_present), 6):
        axes_flat[idx].set_visible(False)
    ordered = [c for c in display_order if c in all_handles]
    fig.legend([all_handles[c] for c in ordered], ordered, title=legend,
               loc="lower right", bbox_to_anchor=(0.98, 0.02), fontsize=9)
    fig.suptitle(f"{legend} by Observance — Faceted by Age ({scope_label})",
                 fontsize=15, y=1.01)
    plt.tight_layout(); _save(fig, f"{num}_observance_{col}_by_age")

    # Companion line graph
    avg_score_lines(df, "observance", score_col, "age", ages_present,
                    line_title, f"{num}b_{col}_by_observance_age_lines",
                    f"Avg. ({ylab})", group_order=OBS_ORDER, ylim=ylim,
                    yticks=list(range(1, int(ylim[1]) + 1)) if ylim[1] <= 7 else None,
                    yticklabels=list(EDU_SHORT.values()) if col == "education" else None)

# ── 17–20: Birth region analyses ───────────────────────────────────────

print("17–20: Birth region")
bar_chart(df["birth_region"], "Birth Region", "Region", "17_birth_region", order=REGION_ORDER)
pie_chart(df["birth_region"], "Birth Region", "17_birth_region_pie")

for col, col_order, cmap, legend, num, title in [
    ("observance", OBS_ORDER, "Set2", "Observance", "18", "Observance by Birth Region"),
    ("education", EDU_ORDER, "tab20", "Education", "19", "Education by Birth Region"),
    ("farsi", FARSI_ORDER, "RdYlGn", "Farsi Level", "20", "Farsi by Birth Region"),
]:
    ct = _reindex(pd.crosstab(df["birth_region"], df[col]), REGION_ORDER, col_order)
    ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
    stacked_bar(ct_norm, title, f"{num}_{col}_by_birth_region",
                cmap=cmap, ylabel="Percentage", pct=True, legend_title=legend)
    lmap = EDU_SHORT if col == "education" else None
    line_chart(ct_norm, title, f"{num}b_{col}_by_birth_region_lines",
               col_order, label_map=lmap, xlabel="Birth Region")

# Avg score lines for education and farsi by birth region
for score, label, name_suf, color_idx, ylim in [
    ("edu_num", "Education (1=<HS → 6=Doctorate)", "19b_education_by_birth_region_avg", 2, (0.5, 6.5)),
    ("farsi_num", "Farsi (1=None → 4=Fluent)", "20b_farsi_by_birth_region_avg", 0, (0.8, 4.2)),
]:
    sub = df[df["birth_region"].isin(REGION_ORDER) & df[score].notna()]
    means = sub.groupby("birth_region")[score].mean().reindex(REGION_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(means.index, means.values, "o-", color=LINE_COLORS[color_idx], linewidth=2.5, markersize=9)
    ax.set_title(f"Avg. {label} by Birth Region ({scope_label})")
    ax.set_ylabel(f"Avg. {label}"); ax.set_ylim(ylim); ax.grid(True, alpha=0.3)
    for i, v in enumerate(means.values):
        ax.text(i, v + (ylim[1] - ylim[0]) * 0.03, f"{v:.2f}", ha="center", fontsize=11)
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    _save(fig, name_suf)

# ── 21: Community membership ───────────────────────────────────────────

print("21: Membership")
mem = df[df["is_member"].isin(["Yes", "No"])]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for ax, grp_col, order, color, title in [
    (ax1, "age", AGE_ORDER, "steelblue", "by Age"),
    (ax2, "observance", OBS_ORDER, "darkorange", "by Observance"),
]:
    ct = pd.crosstab(mem[grp_col], mem["is_member"]).reindex(
        index=[o for o in order if o in mem[grp_col].values])
    pct = ct["Yes"] / ct.sum(axis=1) * 100
    pct.plot.bar(ax=ax, color=color)
    ax.set_title(f"Membership Rate {title}"); ax.set_ylabel("% Members")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter()); ax.set_ylim(0, 105)
    for i, v in enumerate(pct.values):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
fig.suptitle(f"Community Membership Rates ({scope_label})", fontsize=14)
plt.tight_layout(); _save(fig, "21_membership")

# ── 22: Home language multilingualism ──────────────────────────────────

print("22: Multilingualism")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
nl = df["num_home_langs"].dropna().astype(int)
nl_counts = nl.value_counts().sort_index()
nl_total = nl_counts.sum()
nl_counts.plot.bar(ax=ax1, color=PALETTE[:5])
ax1.set_title("Home Languages Spoken"); ax1.set_xlabel("Count"); ax1.set_ylabel("Respondents")
for i, v in enumerate(nl_counts.values):
    ax1.text(i, v + nl_counts.max() * 0.01, f"{v}\n({v/nl_total*100:.0f}%)", ha="center", fontsize=10)
ax1.tick_params(axis="x", rotation=0)

ct_nla = pd.crosstab(df["age"], df["num_home_langs"].fillna(0).astype(int))
ct_nla = ct_nla.reindex(index=[a for a in AGE_ORDER if a in ct_nla.index])
avg = (ct_nla * ct_nla.columns).sum(axis=1) / ct_nla.sum(axis=1)
avg.plot.bar(ax=ax2, color="teal")
ax2.set_title("Avg. Languages by Age"); ax2.set_ylabel("Avg")
for i, v in enumerate(avg.values):
    ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ax2.tick_params(axis="x", rotation=45)
fig.suptitle(f"Home Language Multilingualism ({scope_label})", fontsize=14)
plt.tight_layout(); _save(fig, "22_multilingualism")

# ── 23: Education × Farsi ─────────────────────────────────────────────

print("23: Education × Farsi")
ct_ef = _reindex(pd.crosstab(df["education"], df["farsi"]), EDU_ORDER, FARSI_ORDER)
ct_ef_norm = ct_ef.div(ct_ef.sum(axis=1), axis=0) * 100
stacked_bar(ct_ef_norm, "Farsi by Education", "23_education_x_farsi",
            cmap="RdYlGn", ylabel="Percentage", pct=True, legend_title="Farsi Level")

# ── 24–25: Birth region by age & US-born vs foreign-born ───────────────

print("24–25: Generational shift")
ct_bra = _reindex(pd.crosstab(df["age"], df["birth_region"]), AGE_ORDER, REGION_ORDER)
ct_bra_norm = ct_bra.div(ct_bra.sum(axis=1), axis=0) * 100
stacked_bar(ct_bra_norm, "Birth Region by Age — Normalized", "24_birth_region_by_age",
            cmap="Paired", ylabel="Percentage", pct=True, legend_title="Birth Region",
            figsize=(12, 6))
line_chart(ct_bra_norm, "Birth Region (%) by Age", "24b_birth_region_by_age_lines",
           REGION_ORDER, xlabel="Age Group")

# US-born vs foreign-born observance
fig, ax = plt.subplots(figsize=(10, 6))
for born, label, c in [(True, "US-Born", LINE_COLORS[2]), (False, "Foreign-Born", LINE_COLORS[3])]:
    sub = df[(df["born_us"] == born) & df["age"].isin(AGE_ORDER) & df["obs_num"].notna()]
    means = sub.groupby("age")["obs_num"].mean().reindex(AGE_ORDER).dropna()
    ax.plot(means.index, means.values, "o-", label=label, color=c, linewidth=2.5, markersize=9)
ax.set_title(f"Avg. Observance: US-Born vs Foreign-Born ({scope_label})")
ax.set_ylabel("Observance (1→4)"); ax.set_ylim(1, 4); ax.grid(True, alpha=0.3); ax.legend()
plt.xticks(rotation=45, ha="right"); plt.tight_layout()
_save(fig, "25_observance_usborn_vs_foreign")

# ── 26: NYC neighborhoods (US only) ───────────────────────────────────

if SCOPE == "us":
    print("26: NYC neighborhoods")
    nyc = df[df["region"] == "New York"]
    nyc_cities = nyc["city"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    nyc_cities.plot.barh(ax=ax, color=PALETTE[: len(nyc_cities)])
    ax.set_title(f"NYC Neighborhoods (Top 15)  [N={len(nyc)}]")
    ax.set_xlabel("Respondents"); ax.invert_yaxis()
    for i, v in enumerate(nyc_cities.values):
        ax.text(v + nyc_cities.max() * 0.01, i,
                f" {v} ({v/len(nyc)*100:.1f}%)", va="center", fontsize=9)
    plt.tight_layout(); _save(fig, "26_nyc_neighborhoods")

# ── 27–34: Occupation analyses ─────────────────────────────────────────

print("27–34: Occupation")
occ_counts = df["occupation"].dropna().value_counts()
occ_sorted = occ_counts.reindex([o for o in OCCUPATION_ORDER if o in occ_counts.index]).dropna()
fig, ax = plt.subplots(figsize=(12, 9))
occ_sorted.plot.barh(ax=ax, color=PALETTE[: len(occ_sorted)])
ax.set_title(f"Occupation Distribution ({scope_label})  [N={occ_counts.sum()}]")
ax.set_xlabel("Respondents"); ax.invert_yaxis()
for i, v in enumerate(occ_sorted.values):
    ax.text(v + occ_sorted.max() * 0.01, i,
            f" {v} ({v/occ_counts.sum()*100:.1f}%)", va="center", fontsize=10)
plt.tight_layout(); _save(fig, "27_occupation_distribution")

# Healthcare vs non-healthcare pie
df_occ = df[df["occupation"].notna() & ~df["occupation"].isin(["Student", "Retired", "Homemaker"])]
hc = df_occ["is_healthcare"].value_counts()
fig, ax = plt.subplots(figsize=(7, 7))
labels = {True: f"Healthcare\n(n={hc.get(True,0)})", False: f"Non-Healthcare\n(n={hc.get(False,0)})"}
_, texts, autotexts = ax.pie(
    hc.values, labels=[labels[k] for k in hc.index], autopct=lambda p: f"{p:.1f}%",
    colors=[LINE_COLORS[2], LINE_COLORS[3]], startangle=140, pctdistance=0.75)
for t in [*texts, *autotexts]:
    t.set_fontsize(13)
ax.set_title(f"Healthcare vs Non-Healthcare ({scope_label})\n(excl. students/retired/homemakers)")
plt.tight_layout(); _save(fig, "28_healthcare_pie")

# Occupation cross-tabs (29–33)
top_occ = occ_counts.head(12).index.tolist()
for col, col_order, cmap, legend, num, title in [
    ("age", AGE_ORDER, "Set2", "Age Group", "29", "Age Composition of Top Occupations"),
    ("observance", OBS_ORDER, "Set2", "Observance", "30", "Observance of Top Occupations"),
    ("education", EDU_ORDER, "tab20", "Education", "31", "Education of Top Occupations"),
    ("farsi", FARSI_ORDER, "RdYlGn", "Farsi Level", "32", "Farsi of Top Occupations"),
    ("birth_region", REGION_ORDER, "Paired", "Birth Region", "33", "Birth Region of Top Occupations"),
]:
    sub = df[df["occupation"].isin(top_occ) & df[col].isin(col_order)]
    ct = pd.crosstab(sub["occupation"], sub[col])
    ct = ct.reindex(columns=[c for c in col_order if c in ct.columns])
    if col == "education":
        ct.columns = [EDU_SHORT.get(c, c) for c in ct.columns]
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
    stacked_barh(ct, title, f"{num}_occupation_by_{col}", cmap=cmap, legend_title=legend)

# Healthcare rate by age & observance
df_working = df[df["occupation"].notna() & ~df["occupation"].isin(["Student", "Retired", "Homemaker"])]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
for ax, grp, order, color, title in [
    (ax1, "age", AGE_ORDER, LINE_COLORS[0], "by Age"),
    (ax2, "observance", OBS_ORDER, LINE_COLORS[2], "by Observance"),
]:
    rate = df_working.groupby(grp)["is_healthcare"].mean().reindex(order).dropna() * 100
    ax.plot(rate.index, rate.values, "o-", color=color, linewidth=2.5, markersize=9)
    ax.set_title(f"Healthcare Rate {title}"); ax.set_ylabel("% Healthcare")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter()); ax.grid(True, alpha=0.3)
    for i, v in enumerate(rate.values):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
fig.suptitle(f"Healthcare Employment Rates ({scope_label})", fontsize=16)
plt.tight_layout(); _save(fig, "34_healthcare_rate")

# ── 35–39: Student-adjusted analyses ───────────────────────────────────

print("35–39: Student-adjusted")
occ_ns = df[df["occupation"].notna() & (df["occupation"] != "Student")]["occupation"].value_counts()
occ_ns_sorted = occ_ns.reindex(
    [o for o in OCCUPATION_ORDER if o in occ_ns.index and o != "Student"]).dropna()
fig, ax = plt.subplots(figsize=(12, 9))
occ_ns_sorted.plot.barh(ax=ax, color=PALETTE[: len(occ_ns_sorted)])
ax.set_title(f"Occupations excl. Students ({scope_label})  [N={occ_ns.sum()}]")
ax.set_xlabel("Respondents"); ax.invert_yaxis()
for i, v in enumerate(occ_ns_sorted.values):
    ax.text(v + occ_ns_sorted.max() * 0.01, i,
            f" {v} ({v/occ_ns.sum()*100:.1f}%)", va="center", fontsize=10)
plt.tight_layout(); _save(fig, "35_occupation_no_students")

# Youth side-by-side
df_youth = df[df["age"].isin(["Under 20", "20-29"]) & df["occupation"].notna()]
df_youth_working = df_youth[~df_youth["occupation"].isin(["Student", "Retired", "Homemaker"])]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
for ax, sub, title_suf in [(ax1, df_youth, "All"), (ax2, df_youth_working, "Workers Only")]:
    top = sub["occupation"].value_counts().head(12)
    total = len(sub)
    top.plot.barh(ax=ax, color=PALETTE[: len(top)])
    ax.set_title(f"Under 30 — {title_suf} (N={total})"); ax.set_xlabel("Count"); ax.invert_yaxis()
    for i, v in enumerate(top.values):
        ax.text(v + top.max() * 0.01, i, f" {v} ({v/total*100:.0f}%)", va="center", fontsize=10)
fig.suptitle(f"Youth Occupations ({scope_label})", fontsize=15)
plt.tight_layout(); _save(fig, "36_youth_occupations")

# Occupation by age excl. students
top_ns = [o for o in occ_ns.head(12).index if o != "Student"][:12]
sub_ns = df[df["occupation"].isin(top_ns) & df["age"].isin(AGE_ORDER)]
ct_ns = pd.crosstab(sub_ns["occupation"], sub_ns["age"]).reindex(
    columns=[a for a in AGE_ORDER if a in sub_ns["age"].values])
ct_ns = ct_ns.loc[ct_ns.sum(axis=1).sort_values(ascending=False).index]
stacked_barh(ct_ns, "Age of Top Occupations (excl. Students)", "37_occupation_by_age_no_students",
             cmap="Set2", legend_title="Age Group")

# Healthcare rate: with vs without students
fig, ax = plt.subplots(figsize=(10, 6))
df_has_occ = df[df["occupation"].notna() & df["age"].isin(AGE_ORDER)]
df_wk = df_has_occ[~df_has_occ["occupation"].isin(["Student", "Retired", "Homemaker"])]
for sub, label, color, ls, ms in [
    (df_has_occ, "All (students=non-HC)", LINE_COLORS[5], "--", 7),
    (df_wk, "Workers only", LINE_COLORS[0], "-", 9),
]:
    rate = sub.groupby("age")["is_healthcare"].mean().reindex(AGE_ORDER).dropna() * 100
    ax.plot(rate.index, rate.values, f"o{ls}", label=label, color=color,
            linewidth=2.5 if ls == "-" else 2, markersize=ms, alpha=1 if ls == "-" else 0.7)
wk_rate = df_wk.groupby("age")["is_healthcare"].mean().reindex(AGE_ORDER).dropna() * 100
for i, v in enumerate(wk_rate.values):
    ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=10, color=LINE_COLORS[0])
ax.set_title(f"Healthcare Rate by Age ({scope_label})")
ax.set_ylabel("% Healthcare"); ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
_save(fig, "38_healthcare_students_comparison")

# Youth workers: observance by occupation
df_yw = df_youth_working[df_youth_working["observance"].isin(OBS_ORDER)]
if len(df_yw) > 20:
    top_yw = df_yw["occupation"].value_counts().head(8).index
    ct_yw = pd.crosstab(df_yw[df_yw["occupation"].isin(top_yw)]["occupation"],
                        df_yw[df_yw["occupation"].isin(top_yw)]["observance"])
    ct_yw = ct_yw.reindex(columns=[o for o in OBS_ORDER if o in ct_yw.columns])
    ct_yw = ct_yw.loc[ct_yw.sum(axis=1).sort_values(ascending=False).index]
    stacked_barh(ct_yw, "Observance — Under 30 Workers", "39_youth_observance_by_occ",
                 cmap="Set2", legend_title="Observance")

# ── 40–43: Deviation from baseline ────────────────────────────────────

print("40–43: Deviation analysis")

METRIC_DEFS = [
    ("farsi_num", "Farsi Proficiency\n(1–4 scale)"),
    ("edu_num", "Education Level\n(1–6 scale)"),
    ("obs_num", "Observance Level\n(1–4 scale)"),
    ("hc_pct", "% in Healthcare"),
    ("nyc_born_pct", "% Born in NYC"),
]


def _profile(sub):
    return {
        "farsi_num": sub["farsi_num"].mean(),
        "edu_num": sub["edu_num"].mean(),
        "obs_num": sub["obs_num"].mean(),
        "hc_pct": sub["is_healthcare"].mean() * 100 if sub["is_healthcare"].notna().any() else np.nan,
        "nyc_born_pct": sub["birth_region"].eq("New York City").mean() * 100
                        if sub["birth_region"].notna().any() else np.nan,
    }


def _deviation_panel(ax, baseline, subgroups, metrics, colors):
    y_pos = np.arange(len(metrics))
    h = 0.75 / max(len(subgroups), 1)
    for j, (name, prof, n_grp) in enumerate(subgroups):
        vals = [prof[m] - baseline[m] if pd.notna(baseline[m]) and pd.notna(prof[m]) else 0
                for m, _ in metrics]
        ax.barh(y_pos + j * h - 0.375 + h / 2, vals, h * 0.9,
                label=f"{name} (n={n_grp})", color=colors[j % len(colors)], alpha=0.85)
    ax.set_yticks(y_pos); ax.set_yticklabels([lbl for _, lbl in metrics], fontsize=10)
    ax.axvline(0, color="black", linewidth=1); ax.grid(True, axis="x", alpha=0.2)


def _deviation_faceted(title, name, group_fn, metrics, color_fn):
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    axes_flat = axes.flatten()
    for idx, age in enumerate(ages_present):
        ax = axes_flat[idx]
        sub = df[df["age"] == age]
        baseline = _profile(sub)
        subgroups, colors = group_fn(sub)
        _deviation_panel(ax, baseline, subgroups, metrics, colors)
        ax.set_title(f"Age {age} (n={len(sub)})", fontsize=12)
        ax.set_xlabel("Difference from avg" if idx >= 3 else "")
    for idx in range(len(ages_present), 6):
        axes_flat[idx].set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.02), fontsize=9)
    fig.suptitle(f"{title} ({scope_label})", fontsize=14, y=1.01)
    plt.tight_layout(); _save(fig, name)


metrics_no_obs = [(m, l) for m, l in METRIC_DEFS if m != "obs_num"]
metrics_no_nyc = [(m, l) for m, l in METRIC_DEFS if m != "nyc_born_pct"]

_deviation_faceted(
    "Observance Subgroups vs Age-Group Average", "40_deviation_by_observance",
    lambda sub: (
        [(o, _profile(sub[sub["observance"] == o]), (sub["observance"] == o).sum())
         for o in OBS_ORDER if (sub["observance"] == o).sum() >= 5],
        [LINE_COLORS[OBS_ORDER.index(o)] for o in OBS_ORDER
         if (sub["observance"] == o).sum() >= 5]),
    metrics_no_obs, None)

_deviation_faceted(
    "Birth Region Subgroups vs Age-Group Average", "41_deviation_by_birth_region",
    lambda sub: (
        [(br, _profile(sub[sub["birth_region"] == br]), (sub["birth_region"] == br).sum())
         for br in ["Central Asia / FSU", "New York City"]
         if (sub["birth_region"] == br).sum() >= 5],
        [LINE_COLORS[4], LINE_COLORS[5]]),
    metrics_no_nyc, None)

# 42: 30-39 deep dive
sub_ref = df[df["age"] == "30-39"]
baseline_ref = _profile(sub_ref)
sg42 = []
for obs in OBS_ORDER:
    grp = sub_ref[sub_ref["observance"] == obs]
    if len(grp) >= 5:
        sg42.append((obs, _profile(grp), len(grp)))
for br in ["Central Asia / FSU", "New York City"]:
    grp = sub_ref[sub_ref["birth_region"] == br]
    if len(grp) >= 5:
        sg42.append((f"Born: {br}", _profile(grp), len(grp)))
fig, ax = plt.subplots(figsize=(12, 7))
_deviation_panel(ax, baseline_ref, sg42, metrics_no_nyc, LINE_COLORS)
ax.set_xlabel("Difference from 30-39 Average", fontsize=12)
ax.set_title(f"30-39: Subgroup Deviations ({scope_label}), baseline n={len(sub_ref)}", fontsize=14)
ax.legend(loc="best", fontsize=9); plt.tight_layout()
_save(fig, "42_baseline_30_39")

# 43: All ages vs overall
overall = _profile(df)
sg43 = [(age, _profile(df[df["age"] == age]), (df["age"] == age).sum()) for age in ages_present]
fig, ax = plt.subplots(figsize=(12, 7))
_deviation_panel(ax, overall, sg43, METRIC_DEFS, LINE_COLORS)
ax.set_xlabel("Difference from Overall Average", fontsize=12)
ax.set_title(f"Age Group Deviations from Overall ({scope_label})", fontsize=14)
ax.legend(title="Age", loc="best", fontsize=9); plt.tight_layout()
_save(fig, "43_age_vs_overall")

# ── 44–47: Occupation × observance ────────────────────────────────────

print("44–47: Occupation × Observance")
df_wo = df[df["occupation"].notna() &
           ~df["occupation"].isin(["Student", "Retired", "Homemaker"]) &
           df["observance"].isin(OBS_ORDER)]
obs_with_n = [o for o in OBS_ORDER if (df_wo["observance"] == o).sum() >= 15]

# 44: Faceted occupation panels
n_panels = len(obs_with_n)
fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 8), sharey=False)
if n_panels == 1:
    axes = [axes]
for idx, obs in enumerate(obs_with_n):
    ax = axes[idx]
    sub = df_wo[df_wo["observance"] == obs]
    top = sub["occupation"].value_counts().head(10)
    top.plot.barh(ax=ax, color=LINE_COLORS[OBS_ORDER.index(obs)])
    ax.set_title(f"{obs}\n(n={len(sub)})", fontsize=13); ax.set_xlabel("Count"); ax.invert_yaxis()
    for i, v in enumerate(top.values):
        ax.text(v + top.max() * 0.02, i, f" {v} ({v/len(sub)*100:.0f}%)", va="center", fontsize=10)
fig.suptitle(f"Top Occupations by Observance — Workers ({scope_label})", fontsize=15, y=1.01)
plt.tight_layout(); _save(fig, "44_occupation_by_observance_faceted")

# 45: Rate comparison grid
KEY_CATS = [
    ("Healthcare", HEALTHCARE_CATS), ("Physician", ["Physician"]),
    ("Nurse / NP", ["Nurse / NP"]), ("Pharmacist", ["Pharmacist"]),
    ("Dentist", ["Dentist / Dental"]), ("Barber / Beauty", ["Barber / Beauty"]),
    ("Business Owner", ["Business Owner"]), ("Accounting", ["Accounting / CPA"]),
    ("Law / Legal", ["Law / Legal"]), ("Education", ["Education"]),
    ("Tech / IT", ["Tech / IT"]), ("Real Estate", ["Real Estate"]),
]
fig, axes = plt.subplots(3, 4, figsize=(13, 10))
for idx, (cat_name, cat_list) in enumerate(KEY_CATS):
    ax = axes.flatten()[idx]
    rates = [df_wo[df_wo["observance"] == o]["occupation"].isin(cat_list).mean() * 100
             for o in obs_with_n]
    ax.bar(range(len(obs_with_n)), rates,
           color=[LINE_COLORS[OBS_ORDER.index(o)] for o in obs_with_n])
    ax.set_xticks(range(len(obs_with_n)))
    ax.set_xticklabels([o.replace("-", "-\n") for o in obs_with_n], fontsize=8)
    ax.set_title(cat_name, fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    for i, v in enumerate(rates):
        ax.text(i, v + 0.5, f"{v:.0f}%", ha="center", fontsize=9)
fig.suptitle(f"Occupation Rate by Observance — Workers ({scope_label})", fontsize=14, y=1.01)
plt.tight_layout(); _save(fig, "45_occupation_rate_by_observance")

# 46: Observance mix within each occupation (with population reference)
pop_dist = df_wo["observance"].value_counts(normalize=True).reindex(OBS_ORDER).fillna(0) * 100
top_for_obs = df_wo["occupation"].value_counts().head(14).index
ct46 = pd.crosstab(df_wo[df_wo["occupation"].isin(top_for_obs)]["occupation"],
                   df_wo[df_wo["occupation"].isin(top_for_obs)]["observance"])
ct46 = ct46.reindex(columns=[o for o in OBS_ORDER if o in ct46.columns])
ct46 = ct46.loc[ct46.sum(axis=1).sort_values(ascending=False).index]
pop_row = pd.DataFrame([pop_dist.reindex(ct46.columns).values],
                       columns=ct46.columns, index=["▸ POPULATION AVG"])
ct46_full = pd.concat([pop_row, ct46])
ct46_norm = ct46_full.div(ct46_full.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(13, 8))
ct46_norm.plot.barh(ax=ax, stacked=True, colormap="Set2")
ax.set_title(f"Observance Within Occupations ({scope_label})")
ax.set_xlabel("Percentage"); ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Observance", bbox_to_anchor=(1.02, 1), loc="upper left"); ax.invert_yaxis()
for lbl in ax.get_yticklabels():
    if "POPULATION" in lbl.get_text():
        lbl.set_fontweight("bold"); lbl.set_color("red")
plt.tight_layout(); _save(fig, "46_observance_within_occupation")

# 47: Healthcare subcategories by observance
df_hc = df_wo[df_wo["occupation"].isin(HEALTHCARE_CATS)]
hc_present = [c for c in HEALTHCARE_CATS if (df_hc["occupation"] == c).sum() >= 5]
if len(hc_present) >= 3:
    ct47 = pd.crosstab(df_hc["occupation"], df_hc["observance"])
    ct47 = ct47.reindex(index=[c for c in hc_present if c in ct47.index],
                        columns=[o for o in OBS_ORDER if o in ct47.columns])
    ct47 = ct47.loc[ct47.sum(axis=1).sort_values(ascending=False).index]
    ct47_norm = ct47.div(ct47.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(11, 6))
    ct47_norm.plot.barh(ax=ax, stacked=True, colormap="Set2")
    ax.set_title(f"Observance in Healthcare Subcategories ({scope_label})")
    ax.set_xlabel("Percentage"); ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(title="Observance", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.invert_yaxis()
    for i, cat in enumerate(ct47.index):
        ax.text(101, i, f" n={int(ct47.loc[cat].sum())}", va="center", fontsize=10)
    plt.tight_layout(); _save(fig, "47_healthcare_by_observance")

# ── Summary statistics ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"SUMMARY — {scope_label}")
print("=" * 60)
for col in ["age", "education", "observance", "farsi", "is_member", "birth_region", "occupation"]:
    print(f"\n--- {col.upper()} ---")
    print(df[col].value_counts().to_string())
    print(f"  (missing: {df[col].isna().sum()})")

total = len(list(OUTPUT_DIR.glob("*.png")))
print(f"\n{'='*60}\nAll {total} charts → {OUTPUT_DIR}")
