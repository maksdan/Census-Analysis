import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path
from collections import Counter

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.facecolor': 'white',
})

LINE_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4',
               '#f032e6', '#bfef45', '#000075', '#dcbeff']

SCOPE = sys.argv[1] if len(sys.argv) > 1 else "us"  # "us" or "global"
BASE_DIR = Path("/Users/danielmaksumov/Bukharian Census Analysis")
OUTPUT_DIR = BASE_DIR / "charts" / SCOPE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PALETTE = plt.cm.tab20.colors

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════════
df = pd.read_csv(BASE_DIR / "Bukharian Survey - Raw Data Aug 2021.csv", low_memory=False)
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
df.columns = df.columns.str.strip()

# ── Observance ──────────────────────────────────────────────────────────
df['observance'] = df['observance'].replace({
    'Традиционный': 'Traditional',
    'Строго соблюдающий': 'Very Observant',
    'Не соблюдающий': 'Non-observant',
    'דתי': 'Observant',
    'מסורתי': 'Traditional',
    'לא שומר תורה ומצוות': 'Non-observant',
    'No Answer Provided': np.nan,
})
OBS_ORDER = ['Non-observant', 'Traditional', 'Observant', 'Very Observant']

# ── Farsi ───────────────────────────────────────────────────────────────
df['farsi'] = df['farsi'].replace({
    'с небольшими ограничениями': 'Limited Proficiency',
    'В совершенстве': 'Fluent',
    'понимаю только названия еды и нецензурные слова': 'Only know food names and swear words',
    'не знаю вообще': "Don't know it at all",
    'בינונית': 'Limited Proficiency',
    'אני מבין רק את שמות האוכל והמילים המגונות': 'Only know food names and swear words',
    'מצויין': 'Fluent',
    'בכלל לא יודע': "Don't know it at all",
    'No Answer Provided': np.nan,
})
FARSI_ORDER = [
    "Don't know it at all",
    'Only know food names and swear words',
    'Limited Proficiency',
    'Fluent',
]

# ── Education ───────────────────────────────────────────────────────────
df['education'] = df['education'].replace({
    'окончил(а) институт': 'Bachelors Degree',
    'техникум': 'Associates Degree',
    'среднее': 'High School diploma or GED',
    'кандидат наук': 'Masters Degree',
    'доктор наук': 'Doctorate or PhD',
    'неполное среднее': 'Some High School',
    'תואר ראשון': 'Bachelors Degree',
    'מכללה טכנית': 'Associates Degree',
    'תואר שני': 'Masters Degree',
    'בית ספר תיכון': 'High School diploma or GED',
    'Doctorate or PhD': 'Doctorate or PhD',
    'No Answer Provided': np.nan,
})
EDU_ORDER = [
    'Some High School', 'High School diploma or GED', 'Associates Degree',
    'Bachelors Degree', 'Masters Degree', 'Doctorate or PhD',
]
EDU_SHORT = {
    'Some High School': '<HS',
    'High School diploma or GED': 'HS/GED',
    'Associates Degree': 'Associates',
    'Bachelors Degree': 'Bachelors',
    'Masters Degree': 'Masters',
    'Doctorate or PhD': 'Doctorate',
}

# ── Age ─────────────────────────────────────────────────────────────────
df['age'] = df['age'].replace({'60 +': '60+'})
AGE_ORDER = ['Under 20', '20-29', '30-39', '40-49', '50-59', '60+']

# ── Membership ──────────────────────────────────────────────────────────
df['is_member'] = df['is_member'].replace({
    'Да': 'Yes', 'כן': 'Yes', 'Нет': 'No', 'לא': 'No',
})

# ── Vocation → occupation category ──────────────────────────────────────
def categorize_vocation(val):
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    if v in ('-', 'n/a', 'no answer provided', '', 'none', '-none-'):
        return np.nan

    # Healthcare — physicians & dentists
    if any(k in v for k in ['physician', 'doctor', 'md ', 'surgeon', 'врач',
                             'pediatric', 'internist', 'cardiolog', 'urolog',
                             'anesthesi', 'radiolog', 'psychiatr', 'neurolog',
                             'oncolog', 'dermatolog', 'medical director']):
        return 'Physician'
    if any(k in v for k in ['dentist', 'dental', 'orthodon', 'endodont',
                             'periodon', 'oral surg', 'стоматолог']):
        return 'Dentist / Dental'
    if any(k in v for k in ['pharmacist', 'pharmacy', 'фармацевт']):
        return 'Pharmacist'
    if any(k in v for k in ['optometr', 'optician', 'eye doctor']):
        return 'Optometrist / Optician'

    # Healthcare — nursing & allied health
    if any(k in v for k in ['nurse', 'rn', 'nursing', 'медсестра', 'медбрат']):
        return 'Nurse / NP'
    if any(k in v for k in ['physician assistant', ' pa ', 'pa-c']) or v in ('pa',):
        return 'Physician Assistant'
    if any(k in v for k in ['occupational therap', 'ot ', 'cota']) or v in ('ot',):
        return 'Occupational Therapist'
    if any(k in v for k in ['physical therap', 'pt ', 'physiotherap']) or v in ('pt',):
        return 'Physical Therapist'
    if any(k in v for k in ['speech', 'slp', 'speech pathol', 'логопед']):
        return 'Speech Therapist'
    if any(k in v for k in ['respiratory', 'hemodialysis', 'medical assist',
                             'medical biller', 'medical coder', 'hha',
                             'home health', 'medical tech', 'phlebotom',
                             'x-ray', 'sonograph', 'ultrasound',
                             'behavior analyst', 'bcba']):
        return 'Other Allied Health'

    # Students
    if any(k in v for k in ['student', 'school', 'studying', 'in college',
                             'учусь', 'студент', 'учеба']):
        return 'Student'

    # Law
    if any(k in v for k in ['lawyer', 'attorney', 'paralegal', 'legal',
                             'адвокат', 'юрист']):
        return 'Law / Legal'

    # Finance / Accounting
    if any(k in v for k in ['accountant', 'accounting', 'cpa', 'bookkeep',
                             'auditor', 'бухгалтер', 'tax preparer']):
        return 'Accounting / CPA'
    if any(k in v for k in ['finance', 'banking', 'financial', 'investment',
                             'mortgage', 'loan', 'credit']):
        return 'Finance / Banking'

    # Real estate
    if any(k in v for k in ['real estate', 'realtor', 'realty', 'broker']):
        return 'Real Estate'

    # Tech / Engineering
    if any(k in v for k in ['software', 'developer', 'programmer', 'coding',
                             'it ', 'cyber', 'data', 'web ',
                             'программист', 'computer', 'devops', 'qa',
                             'информатик', 'tech lead', 'full stack']):
        return 'Tech / IT'
    if any(k in v for k in ['engineer', 'engineering', 'инженер', 'архитект',
                             'architect']) and 'software' not in v:
        return 'Engineering'

    # Education
    if any(k in v for k in ['teacher', 'professor', 'education', 'tutor',
                             'educator', 'pedagog', 'педагог', 'преподав',
                             'учитель', 'instructor', 'special ed']):
        return 'Education'

    # Business / Management
    if any(k in v for k in ['business', 'entrepreneur', 'ceo', 'owner',
                             'self employ', 'self-employ', 'предприним',
                             'бизнес']):
        return 'Business Owner'
    if any(k in v for k in ['manager', 'director', 'supervisor', 'менеджер',
                             'управля', 'coordinator', 'администрат']):
        return 'Management'

    # Sales
    if any(k in v for k in ['sales', 'продавец', 'продажи']):
        return 'Sales'

    # Barber / Beauty
    if any(k in v for k in ['barber', 'парикмахер', 'hairdress', 'hairstyl',
                             'cosmetol', 'beauty', 'nail tech', 'estheti',
                             'makeup']):
        return 'Barber / Beauty'

    # Jewelry / Trades
    if any(k in v for k in ['jewel', 'ювелир']):
        return 'Jewelry'
    if any(k in v for k in ['cobbler', 'tailor', 'портной', 'сапожник',
                             'shoemaker', 'seamstress', 'швея']):
        return 'Tailor / Cobbler'
    if any(k in v for k in ['contractor', 'construction', 'electrician',
                             'plumb', 'hvac', 'строител', 'mechanic',
                             'welding', 'carpent']):
        return 'Construction / Trades'
    if any(k in v for k in ['driver', 'водитель', 'trucker', 'cdl', 'taxi',
                             'uber', 'delivery']):
        return 'Driver / Transport'

    # Retired / Homemaker
    if any(k in v for k in ['retire', 'пенсионер', 'пенсионерка']):
        return 'Retired'
    if any(k in v for k in ['house wife', 'housewife', 'homemaker',
                             'stay at home', 'домохозяйка', 'домохоз']):
        return 'Homemaker'

    # Food / Hospitality
    if any(k in v for k in ['cook', 'chef', 'restaurant', 'повар', 'baker',
                             'кулинар', 'catering', 'food']):
        return 'Food / Hospitality'

    # Arts / Media
    if any(k in v for k in ['photograph', 'musician', 'музыкант', 'artist',
                             'дизайн', 'design', 'graphic', 'film',
                             'video', 'media', 'marketing', 'журнали',
                             'writer', 'journal']):
        return 'Arts / Media / Marketing'

    # Social work / Nonprofit / Government / Rabbi
    if any(k in v for k in ['social work', 'rabbi', 'раввин', 'clergy',
                             'nonprofit', 'community', 'government',
                             'counselor', 'психолог', 'therapist']):
        return 'Social Work / Community'

    return 'Other'

df['occupation'] = df['vocation'].apply(categorize_vocation)

OCCUPATION_ORDER = [
    'Physician', 'Dentist / Dental', 'Pharmacist', 'Nurse / NP',
    'Physician Assistant', 'Occupational Therapist', 'Physical Therapist',
    'Speech Therapist', 'Optometrist / Optician', 'Other Allied Health',
    'Law / Legal', 'Accounting / CPA', 'Finance / Banking', 'Real Estate',
    'Tech / IT', 'Engineering', 'Education', 'Business Owner', 'Management',
    'Sales', 'Barber / Beauty', 'Jewelry', 'Tailor / Cobbler',
    'Construction / Trades', 'Driver / Transport',
    'Food / Hospitality', 'Arts / Media / Marketing',
    'Social Work / Community', 'Student', 'Retired', 'Homemaker', 'Other',
]

HEALTHCARE_CATS = [
    'Physician', 'Dentist / Dental', 'Pharmacist', 'Nurse / NP',
    'Physician Assistant', 'Occupational Therapist', 'Physical Therapist',
    'Speech Therapist', 'Optometrist / Optician', 'Other Allied Health',
]

# ── Current location ────────────────────────────────────────────────────
df['current_loc'] = df['current_loc'].replace({
    'США': 'United States', 'ארה&quot;ב': 'United States',
    'Израиль': 'Israel', 'ישראל': 'Israel', 'В Израиле': 'Israel',
    'Канада': 'Canada',
    'Средняя Азиа': 'Central Asia', 'Central Asia': 'Central Asia',
    'Евросоюз': 'European Union',
    'Россия/Украина': 'Russia/Ukraine',
    'Другое': 'Other',
    'No Answer Provided': np.nan,
})

# ── Birth city normalization ────────────────────────────────────────────
def normalize_birth_city(val):
    if pd.isna(val):
        return val
    s = str(val).strip()
    low = s.lower().replace(',', '').replace('.', '').replace('-', '')
    lookup = {
        'tashkent': 'Tashkent', 'ташкент': 'Tashkent', 'ташкентх': 'Tashkent',
        'ташкеит': 'Tashkent', 'ташкенте': 'Tashkent', 'taskent': 'Tashkent',
        'tashekent': 'Tashkent', 'taschkent': 'Tashkent', 'toshkent': 'Tashkent',
        'טשכנט': 'Tashkent',
        'samarkand': 'Samarkand', 'самарканд': 'Samarkand', 'camarkand': 'Samarkand',
        'smarkand': 'Samarkand', 'sanarkand': 'Samarkand', 'samarqand': 'Samarkand',
        'סמרקנד': 'Samarkand', 'סמאקנד': 'Samarkand',
        'samarkand uzbekistan': 'Samarkand', 'samarkand ussr': 'Samarkand',
        'samarkand uz': 'Samarkand', 'uzbekistan samarkand': 'Samarkand',
        'uzbekistan samarkakd': 'Samarkand',
        'bukhara': 'Bukhara', 'бухара': 'Bukhara', 'бухаре': 'Bukhara',
        'buhara': 'Bukhara', 'buchara': 'Bukhara', 'bychara': 'Bukhara',
        'byshara': 'Bukhara', 'buharara': 'Bukhara', 'בוכרה': 'Bukhara',
        'bukhara uzbekistan': 'Bukhara',
        'dushanbe': 'Dushanbe', 'душанбе': 'Dushanbe', 'דושנבה': 'Dushanbe',
        'dushambe': 'Dushanbe', 'dushambeh': 'Dushanbe', 'duschanbe': 'Dushanbe',
        'paishanbe': 'Dushanbe', 'paishambe': 'Dushanbe',
        'dushanbe tadjikistan': 'Dushanbe', 'dushanbe tadzhikistan': 'Dushanbe',
        'tadjistan /dushanbe': 'Dushanbe',
        'fergana': 'Fergana', 'фергана': 'Fergana', 'ferghana': 'Fergana',
        'fergana uzbekistan': 'Fergana',
        'узбекистан фергана': 'Fergana',
        'andizhan': 'Andijan', 'andijan': 'Andijan', 'andizan': 'Andijan',
        'andijon': 'Andijan', 'андижан': 'Andijan', 'undizan': 'Andijan',
        'andezhan': 'Andijan', 'andijhan uzbekistan': 'Andijan',
        'andizhanuzbekistan': 'Andijan',
        'kokand': 'Kokand', 'коканд': 'Kokand', 'коканд узбекистан': 'Kokand',
        'kokand uzbekistan': 'Kokand',
        'navoi': 'Navoi', 'навои': 'Navoi', 'navaii': 'Navoi', 'navai': 'Navoi',
        'navoii': 'Navoi', 'novoi': 'Navoi', 'navio': 'Navoi',
        'kattakurgan': 'Kattakurgan', 'katakurgan': 'Kattakurgan',
        'kattaqurghon': 'Kattakurgan', 'kattakurgan': 'Kattakurgan',
        'катакурган': 'Kattakurgan', 'katta kurgan': 'Kattakurgan',
        'margilan': 'Margilan', 'margelan': 'Margilan', 'маргелан': 'Margilan',
        'маргилан': 'Margilan', 'margilan uzbekistan': 'Margilan',
        'chimkent': 'Chimkent', 'чимкент': 'Chimkent', 'chenkent': 'Chimkent',
        'shymkent': 'Chimkent', 'chimkent kazakstan': 'Chimkent',
        'chimkent kazakhstan': 'Chimkent',
        'leninabad': 'Leninabad', 'ленинабад': 'Leninabad',
        'leninbad': 'Leninabad', 'linlabad': 'Leninabad',
        'leninobad': 'Leninabad', 'leninabad tashkent': 'Leninabad',
        'khujand': 'Leninabad', 'khuzhand': 'Leninabad', 'hodgent': 'Leninabad',
        'namangan': 'Namangan', 'наманган': 'Namangan', 'nsmangan': 'Namangan',
        'denau': 'Denau', 'денау': 'Denau',
        'shahrisabz': 'Shahrisabz', 'шахрисабз': 'Shahrisabz',
        'shakrisabz': 'Shahrisabz', 'shaahrisabz': 'Shahrisabz',
        'shachrisabs': 'Shahrisabz', 'shahrisabs': 'Shahrisabz',
        'tel aviv': 'Tel Aviv', 'telaviv': 'Tel Aviv', 'tel aviv israel': 'Tel Aviv',
        'telaviv': 'Tel Aviv',
        'moscow': 'Moscow', 'мoscow': 'Moscow',
        'israel': 'Israel', 'isreal': 'Israel', 'ישראל': 'Israel',
    }
    # NYC area normalization
    nyc_keys = [
        'new york', 'new york city', 'nyc', 'ny', 'new york ny',
        'new york city', 'new york queens', 'new york ny',
        'usa  brooklyn ny', 'new york city', 'manhattan ny',
        'manhattan', 'kew gardens',
    ]
    queens_keys = [
        'queens', 'queens ny', 'queens new york city',
    ]
    fh_keys = [
        'forest hills', 'forest hills ny', 'forest hills',
        'forrest hills', 'forest hulls',
    ]

    if low in lookup:
        return lookup[low]
    if low in nyc_keys:
        return 'New York City'
    if low in queens_keys:
        return 'Queens, NYC'
    if low in fh_keys:
        return 'Forest Hills, NYC'
    if low == 'flushing' or low == 'flushing ny':
        return 'Flushing, NYC'
    if low == 'rego park':
        return 'Rego Park, NYC'
    if low == 'brooklyn':
        return 'Brooklyn, NYC'
    if low == 'corona':
        return 'Corona, NYC'
    if low in ('bayside queens',):
        return 'Queens, NYC'
    if low in ('fresh meadows',):
        return 'Fresh Meadows, NYC'
    return s

df['birth_city'] = df['birth_city'].apply(normalize_birth_city)

# ── Derived columns ────────────────────────────────────────────────────
central_asian_cities = {
    'Tashkent', 'Samarkand', 'Bukhara', 'Dushanbe', 'Fergana', 'Andijan',
    'Kokand', 'Navoi', 'Kattakurgan', 'Margilan', 'Chimkent', 'Leninabad',
    'Namangan', 'Denau', 'Shahrisabz', 'Moscow',
    'Tajikistan', 'Uzbekistan', 'Kazakhstan', 'Tadzhikistan', 'Turkestan',
    'Turkmenistan', 'Bishkek', 'Frunze',
}
israeli_cities = {'Tel Aviv', 'Israel', 'Jerusalem', 'Holon', 'Ramla', 'Ashdod'}

def birth_region(city):
    if pd.isna(city):
        return np.nan
    if city in central_asian_cities:
        return 'Central Asia / FSU'
    if city in israeli_cities or 'israel' in str(city).lower():
        return 'Israel'
    nyc_tags = ['NYC', 'Queens', 'Brooklyn', 'Flushing', 'Forest Hills',
                'Fresh Meadows', 'Rego Park', 'Corona', 'New York']
    if any(tag in str(city) for tag in nyc_tags):
        return 'New York City'
    us_cities = {'Long Island', 'San Diego', 'Cleveland', 'Miami', 'Los Angeles',
                 'United States', 'Cincinnati', 'New Haven'}
    if city in us_cities or 'US' in str(city):
        return 'Other US'
    euro = {'Vienna', 'Berlin', 'Hannover', 'Düsseldorf', 'Odessa', 'Kiev', 'Toronto'}
    if city in euro or 'Canada' in str(city):
        return 'Europe / Other'
    return 'Central Asia / FSU'

df['birth_region'] = df['birth_city'].apply(birth_region)

# Home language count
def count_langs(val):
    if pd.isna(val):
        return np.nan
    return len([l for l in str(val).split('.') if l.strip()])

df['num_home_langs'] = df['home_lang'].apply(count_langs)

# ── Scope filter ────────────────────────────────────────────────────────
if SCOPE == "us":
    df = df[df['country'] == 'US'].copy()
    scope_label = "United States"
else:
    scope_label = "Global"

print(f"Scope: {scope_label} | Respondents: {len(df)}")
print(f"Output: {OUTPUT_DIR}")
print()

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════
def save(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.png", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {name}.png")

def bar_chart(series, title, xlabel, ylabel="Number of Respondents", name=None,
              horizontal=False, order=None, figsize=(10, 6)):
    counts = series.dropna().value_counts()
    if order is not None:
        counts = counts.reindex([o for o in order if o in counts.index]).dropna()
    total = counts.sum()
    fig, ax = plt.subplots(figsize=figsize)
    colors = list(PALETTE[:len(counts)])
    if horizontal:
        counts.plot.barh(ax=ax, color=colors)
        ax.set_xlabel(ylabel); ax.set_ylabel(xlabel)
        ax.invert_yaxis()
        for i, v in enumerate(counts.values):
            pct = v / total * 100
            ax.text(v + counts.max() * 0.01, i, f" {v} ({pct:.1f}%)", va='center', fontsize=11)
    else:
        counts.plot.bar(ax=ax, color=colors)
        ax.set_ylabel(ylabel); ax.set_xlabel(xlabel)
        for i, v in enumerate(counts.values):
            pct = v / total * 100
            ax.text(i, v + counts.max() * 0.01, f"{v}\n({pct:.1f}%)", ha='center', fontsize=10)
    ax.set_title(f"{title} ({scope_label})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save(fig, name or title.lower().replace(' ', '_'))

def pie_chart(series, title, name=None, figsize=(8, 8)):
    counts = series.dropna().value_counts()
    total = counts.sum()
    labels = [f"{lbl}\n(n={v})" for lbl, v in zip(counts.index, counts.values)]
    fig, ax = plt.subplots(figsize=figsize)
    def autopct(pct):
        return f'{pct:.1f}%' if pct >= 3 else ''
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels, autopct=autopct,
        colors=PALETTE[:len(counts)], startangle=140, pctdistance=0.8)
    for t in texts:
        t.set_fontsize(12)
    for t in autotexts:
        t.set_fontsize(12)
    ax.set_title(f"{title} ({scope_label})  [N={total}]")
    plt.tight_layout()
    save(fig, name or title.lower().replace(' ', '_'))

def reindex_ct(ct, row_order=None, col_order=None):
    if row_order:
        ct = ct.reindex(index=[r for r in row_order if r in ct.index])
    if col_order:
        ct = ct.reindex(columns=[c for c in col_order if c in ct.columns])
    return ct.dropna(how='all', axis=0).dropna(how='all', axis=1)


# ═══════════════════════════════════════════════════════════════════════
# 1. AGE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════
print("1. Age Distribution")
bar_chart(df['age'], "Age Distribution of Respondents", "Age Group",
          order=AGE_ORDER, name="01_age_distribution")
pie_chart(df['age'], "Age Distribution", name="01_age_pie")

# ═══════════════════════════════════════════════════════════════════════
# 2. EDUCATION LEVEL
# ═══════════════════════════════════════════════════════════════════════
print("\n2. Education Level")
bar_chart(df['education'], "Education Level", "Education",
          order=EDU_ORDER, name="02_education_bar")
pie_chart(df['education'], "Education Level Distribution", name="02_education_pie")

# ═══════════════════════════════════════════════════════════════════════
# 3. RELIGIOUS OBSERVANCE
# ═══════════════════════════════════════════════════════════════════════
print("\n3. Religious Observance")
bar_chart(df['observance'], "Level of Religious Observance", "Observance Level",
          order=OBS_ORDER, name="03_observance_bar")
pie_chart(df['observance'], "Religious Observance Distribution", name="03_observance_pie")

# ═══════════════════════════════════════════════════════════════════════
# 4. FARSI PROFICIENCY
# ═══════════════════════════════════════════════════════════════════════
print("\n4. Farsi Proficiency")
bar_chart(df['farsi'], "Farsi / Bukharian Language Proficiency",
          "Proficiency Level", order=FARSI_ORDER,
          name="04_farsi_proficiency", horizontal=True, figsize=(10, 5))

# ═══════════════════════════════════════════════════════════════════════
# 5. HOME LANGUAGES
# ═══════════════════════════════════════════════════════════════════════
print("\n5. Home Languages")
home_lang_map = {
    'русский': 'Russian', 'иврит': 'Hebrew', 'עִברִית': 'Hebrew',
    'английский': 'English', 'бухарский': 'Bukharian',
    'רוסית': 'Russian', 'עברית': 'Hebrew', 'אנגלית': 'English',
    'בוכרית': 'Bukharian', 'תורכי': 'Other', 'другой': 'Other',
}
lang_counter = Counter()
for langs in df['home_lang'].dropna():
    for lang in str(langs).split('.'):
        lang = lang.strip()
        if lang:
            lang = home_lang_map.get(lang, lang)
            lang_counter[lang] += 1
lang_df = pd.Series(lang_counter).sort_values(ascending=False)
n_respondents = len(df)
fig, ax = plt.subplots(figsize=(10, 5))
lang_df.plot.bar(ax=ax, color=PALETTE[:len(lang_df)])
ax.set_title(f"Languages Spoken at Home ({scope_label})  [N={n_respondents}, multi-select]")
ax.set_ylabel("Number of Respondents")
ax.set_xlabel("Language")
for i, v in enumerate(lang_df.values):
    pct = v / n_respondents * 100
    ax.text(i, v + lang_df.max() * 0.01, f"{v}\n({pct:.0f}%)", ha='center', fontsize=10)
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
save(fig, "05_home_languages")

# ═══════════════════════════════════════════════════════════════════════
# 6. CURRENT LOCATION
# ═══════════════════════════════════════════════════════════════════════
print("\n6. Current Location")
if SCOPE == "us":
    bar_chart(df['region'].dropna(), "US State Distribution", "State",
              name="06_us_state", horizontal=True, figsize=(10, 8))
else:
    bar_chart(df['current_loc'].dropna(), "Current Country of Residence",
              "Country", name="06_current_location", horizontal=True)

# ═══════════════════════════════════════════════════════════════════════
# 7. BIRTH CITY
# ═══════════════════════════════════════════════════════════════════════
print("\n7. Birth City (Top 20)")
birth_counts = df['birth_city'].dropna().value_counts().head(20)
birth_total = df['birth_city'].dropna().shape[0]
fig, ax = plt.subplots(figsize=(10, 7))
birth_counts.plot.barh(ax=ax, color=PALETTE[:len(birth_counts)])
ax.set_title(f"Birth City — Top 20 ({scope_label})  [N={birth_total}]")
ax.set_xlabel("Number of Respondents")
ax.invert_yaxis()
for i, v in enumerate(birth_counts.values):
    pct = v / birth_total * 100
    ax.text(v + birth_counts.max() * 0.01, i, f" {v} ({pct:.1f}%)", va='center', fontsize=11)
plt.tight_layout()
save(fig, "07_birth_city")

# ═══════════════════════════════════════════════════════════════════════
# 8. COMMUNITY MEMBERSHIP
# ═══════════════════════════════════════════════════════════════════════
print("\n8. Community Membership")
pie_chart(df['is_member'], "Community Organization Membership", name="08_membership")

# ═══════════════════════════════════════════════════════════════════════
# 9. AGE × OBSERVANCE (stacked + normalized)
# ═══════════════════════════════════════════════════════════════════════
print("\n9. Cross-tab: Age × Observance")
ct = reindex_ct(pd.crosstab(df['age'], df['observance']), AGE_ORDER, OBS_ORDER)
fig, ax = plt.subplots(figsize=(11, 6))
ct.plot.bar(ax=ax, stacked=True, colormap='Set2')
ax.set_title(f"Religious Observance by Age Group ({scope_label})")
ax.set_ylabel("Count"); ax.set_xlabel("Age Group")
ax.legend(title="Observance", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
save(fig, "09_age_x_observance")

ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(11, 6))
ct_norm.plot.bar(ax=ax, stacked=True, colormap='Set2')
ax.set_title(f"Religious Observance by Age Group — Normalized ({scope_label})")
ax.set_ylabel("Percentage"); ax.set_xlabel("Age Group")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Observance", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
save(fig, "09_age_x_observance_pct")

# 9b. LINE: % of each observance level across age groups
print("  9b. Line graph: Observance % by Age")
fig, ax = plt.subplots(figsize=(10, 6))
for idx, obs in enumerate(OBS_ORDER):
    if obs in ct_norm.columns:
        ax.plot(ct_norm.index, ct_norm[obs].values, 'o-', label=obs,
                color=LINE_COLORS[idx], linewidth=2.5, markersize=8)
ax.set_title(f"Observance Level (%) by Age Group ({scope_label})")
ax.set_ylabel("% of Age Group"); ax.set_xlabel("Age Group")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(True, alpha=0.3); ax.legend(title="Observance")
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
save(fig, "09b_observance_by_age_lines")

# ═══════════════════════════════════════════════════════════════════════
# 10. AGE × EDUCATION (stacked)
# ═══════════════════════════════════════════════════════════════════════
print("\n10. Cross-tab: Age × Education")
ct2 = reindex_ct(pd.crosstab(df['age'], df['education']), AGE_ORDER, EDU_ORDER)
fig, ax = plt.subplots(figsize=(12, 6))
ct2.plot.bar(ax=ax, stacked=True, colormap='tab20')
ax.set_title(f"Education Level by Age Group ({scope_label})")
ax.set_ylabel("Count"); ax.set_xlabel("Age Group")
ax.legend(title="Education", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
save(fig, "10_age_x_education")

# 10b. LINE: % of each education level across age groups
print("  10b. Line graph: Education % by Age")
ct2_norm = ct2.div(ct2.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(10, 6))
for idx, ed in enumerate(EDU_ORDER):
    if ed in ct2_norm.columns:
        ax.plot(ct2_norm.index, ct2_norm[ed].values, 'o-', label=EDU_SHORT[ed],
                color=LINE_COLORS[idx], linewidth=2.5, markersize=8)
ax.set_title(f"Education Level (%) by Age Group ({scope_label})")
ax.set_ylabel("% of Age Group"); ax.set_xlabel("Age Group")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(True, alpha=0.3); ax.legend(title="Education")
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
save(fig, "10b_education_by_age_lines")

# ═══════════════════════════════════════════════════════════════════════
# 11. AGE × FARSI (normalized)
# ═══════════════════════════════════════════════════════════════════════
print("\n11. Cross-tab: Age × Farsi")
ct3 = reindex_ct(pd.crosstab(df['age'], df['farsi']), AGE_ORDER, FARSI_ORDER)
ct3_norm = ct3.div(ct3.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(12, 6))
ct3_norm.plot.bar(ax=ax, stacked=True, colormap='RdYlGn')
ax.set_title(f"Farsi Proficiency by Age Group — Normalized ({scope_label})")
ax.set_ylabel("Percentage"); ax.set_xlabel("Age Group")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Farsi Level", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
save(fig, "11_age_x_farsi_pct")

# 11b. LINE: % of each Farsi level across age groups
print("  11b. Line graph: Farsi % by Age")
fig, ax = plt.subplots(figsize=(10, 6))
for idx, fl in enumerate(FARSI_ORDER):
    if fl in ct3_norm.columns:
        ax.plot(ct3_norm.index, ct3_norm[fl].values, 'o-', label=fl,
                color=LINE_COLORS[idx], linewidth=2.5, markersize=8)
ax.set_title(f"Farsi Proficiency (%) by Age Group ({scope_label})")
ax.set_ylabel("% of Age Group"); ax.set_xlabel("Age Group")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(True, alpha=0.3); ax.legend(title="Farsi Level")
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
save(fig, "11b_farsi_by_age_lines")

# ═══════════════════════════════════════════════════════════════════════
# 12. OBSERVANCE × EDUCATION (normalized)
# ═══════════════════════════════════════════════════════════════════════
print("\n12. Cross-tab: Observance × Education")
ct4 = reindex_ct(pd.crosstab(df['observance'], df['education']), OBS_ORDER, EDU_ORDER)
ct4_norm = ct4.div(ct4.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(11, 6))
ct4_norm.plot.bar(ax=ax, stacked=True, colormap='tab20')
ax.set_title(f"Education by Observance Level — Normalized ({scope_label})")
ax.set_ylabel("Percentage"); ax.set_xlabel("Observance")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Education", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
save(fig, "12_observance_x_education_pct")

# 12b. LINE: % of each education level across observance levels
print("  12b. Line graph: Education % by Observance")
fig, ax = plt.subplots(figsize=(10, 6))
for idx, ed in enumerate(EDU_ORDER):
    if ed in ct4_norm.columns:
        ax.plot(ct4_norm.index, ct4_norm[ed].values, 'o-', label=EDU_SHORT[ed],
                color=LINE_COLORS[idx], linewidth=2.5, markersize=8)
ax.set_title(f"Education Level (%) by Observance ({scope_label})")
ax.set_ylabel("% of Observance Group"); ax.set_xlabel("Observance Level")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(True, alpha=0.3); ax.legend(title="Education")
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
save(fig, "12b_education_by_observance_lines")

# ═══════════════════════════════════════════════════════════════════════
# 13. OBSERVANCE × FARSI (normalized)
# ═══════════════════════════════════════════════════════════════════════
print("\n13. Cross-tab: Observance × Farsi")
ct5 = reindex_ct(pd.crosstab(df['observance'], df['farsi']), OBS_ORDER, FARSI_ORDER)
ct5_norm = ct5.div(ct5.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(11, 6))
ct5_norm.plot.bar(ax=ax, stacked=True, colormap='RdYlGn')
ax.set_title(f"Farsi Proficiency by Observance — Normalized ({scope_label})")
ax.set_ylabel("Percentage"); ax.set_xlabel("Observance Level")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Farsi Level", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
save(fig, "13_observance_x_farsi_pct")

# 13b. LINE: % of each Farsi level across observance levels
print("  13b. Line graph: Farsi % by Observance")
fig, ax = plt.subplots(figsize=(10, 6))
for idx, fl in enumerate(FARSI_ORDER):
    if fl in ct5_norm.columns:
        ax.plot(ct5_norm.index, ct5_norm[fl].values, 'o-', label=fl,
                color=LINE_COLORS[idx], linewidth=2.5, markersize=8)
ax.set_title(f"Farsi Proficiency (%) by Observance ({scope_label})")
ax.set_ylabel("% of Observance Group"); ax.set_xlabel("Observance Level")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(True, alpha=0.3); ax.legend(title="Farsi Level")
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
save(fig, "13b_farsi_by_observance_lines")

# ═══════════════════════════════════════════════════════════════════════
# 14. FARSI × AGE HEATMAP
# ═══════════════════════════════════════════════════════════════════════
print("\n14. Heatmap: Farsi × Age")
heat = reindex_ct(pd.crosstab(df['farsi'], df['age']), FARSI_ORDER, AGE_ORDER)
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(heat.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(heat.columns))); ax.set_xticklabels(heat.columns, rotation=45, ha='right')
ax.set_yticks(range(len(heat.index))); ax.set_yticklabels(heat.index)
for i in range(len(heat.index)):
    for j in range(len(heat.columns)):
        val = heat.values[i, j]
        color = 'white' if val > heat.values.max() * 0.6 else 'black'
        ax.text(j, i, str(int(val)), ha='center', va='center', fontsize=12, color=color)
fig.colorbar(im, ax=ax, label='Count')
ax.set_title(f"Farsi Proficiency × Age Group ({scope_label})")
plt.tight_layout()
save(fig, "14_farsi_age_heatmap")

# ═══════════════════════════════════════════════════════════════════════
# 15. OBSERVANCE × FARSI — FACETED BY AGE GROUP
# ═══════════════════════════════════════════════════════════════════════
print("\n15. Faceted: Observance × Farsi by Age Group")
ages_present = [a for a in AGE_ORDER if a in df['age'].values]
n = len(ages_present)
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
axes = axes.flatten()
all_handles, all_labels = {}, {}
for idx, age in enumerate(ages_present):
    ax = axes[idx]
    sub = df[df['age'] == age]
    ct_f = reindex_ct(pd.crosstab(sub['observance'], sub['farsi']), OBS_ORDER, FARSI_ORDER)
    if ct_f.empty:
        ax.set_visible(False); continue
    ct_f_norm = ct_f.div(ct_f.sum(axis=1), axis=0) * 100
    ct_f_norm.plot.bar(ax=ax, stacked=True, colormap='RdYlGn', legend=False)
    ax.set_title(f"Age {age} (n={len(sub)})", fontsize=12)
    ax.set_ylabel("%" if idx % 3 == 0 else "")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.tick_params(axis='x', rotation=45)
    for h, l in zip(*ax.get_legend_handles_labels()):
        all_handles[l] = h
for idx in range(n, len(axes)):
    axes[idx].set_visible(False)
ordered_labels = [f for f in FARSI_ORDER if f in all_handles]
fig.legend([all_handles[l] for l in ordered_labels], ordered_labels,
           title="Farsi Proficiency", loc='lower right',
           bbox_to_anchor=(0.98, 0.02), fontsize=9)
fig.suptitle(f"Farsi Proficiency by Observance — Faceted by Age ({scope_label})",
             fontsize=15, y=1.01)
plt.tight_layout()
save(fig, "15_observance_farsi_by_age")

# 15b. LINE GRAPH: Avg Farsi score by observance, one line per age group
print("  15b. Line graph: Avg Farsi by Observance × Age")
farsi_numeric = {"Don't know it at all": 1, 'Only know food names and swear words': 2,
                 'Limited Proficiency': 3, 'Fluent': 4}
df['farsi_num'] = df['farsi'].map(farsi_numeric)
fig, ax = plt.subplots(figsize=(10, 6))
for idx, age in enumerate(ages_present):
    sub = df[(df['age'] == age) & df['observance'].isin(OBS_ORDER) & df['farsi_num'].notna()]
    means = sub.groupby('observance')['farsi_num'].mean().reindex(OBS_ORDER).dropna()
    ax.plot(means.index, means.values, 'o-', label=f"{age} (n={len(sub)})",
            color=LINE_COLORS[idx], linewidth=2.5, markersize=8)
ax.set_title(f"Avg. Farsi Proficiency by Observance Level, by Age ({scope_label})")
ax.set_ylabel("Avg. Farsi Score\n(1=None → 4=Fluent)")
ax.set_xlabel("Observance Level")
ax.set_ylim(0.8, 4.2); ax.grid(True, alpha=0.3)
ax.legend(title="Age Group")
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
save(fig, "15b_farsi_by_observance_age_lines")

# ═══════════════════════════════════════════════════════════════════════
# 16. OBSERVANCE × EDUCATION — FACETED BY AGE GROUP
# ═══════════════════════════════════════════════════════════════════════
print("\n16. Faceted: Observance × Education by Age Group")
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
axes = axes.flatten()
all_handles, all_labels = {}, {}
edu_short_order = [EDU_SHORT[e] for e in EDU_ORDER]
for idx, age in enumerate(ages_present):
    ax = axes[idx]
    sub = df[df['age'] == age]
    ct_e = reindex_ct(pd.crosstab(sub['observance'], sub['education']), OBS_ORDER, EDU_ORDER)
    if ct_e.empty:
        ax.set_visible(False); continue
    ct_e.columns = [EDU_SHORT.get(c, c) for c in ct_e.columns]
    ct_e_norm = ct_e.div(ct_e.sum(axis=1), axis=0) * 100
    ct_e_norm.plot.bar(ax=ax, stacked=True, colormap='tab20', legend=False)
    ax.set_title(f"Age {age} (n={len(sub)})", fontsize=12)
    ax.set_ylabel("%" if idx % 3 == 0 else "")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.tick_params(axis='x', rotation=45)
    for h, l in zip(*ax.get_legend_handles_labels()):
        all_handles[l] = h
for idx in range(n, len(axes)):
    axes[idx].set_visible(False)
ordered_edu = [e for e in edu_short_order if e in all_handles]
fig.legend([all_handles[l] for l in ordered_edu], ordered_edu,
           title="Education", loc='lower right',
           bbox_to_anchor=(0.98, 0.02), fontsize=9)
fig.suptitle(f"Education by Observance — Faceted by Age ({scope_label})",
             fontsize=15, y=1.01)
plt.tight_layout()
save(fig, "16_observance_edu_by_age")

# 16b. LINE GRAPH: Avg Education score by observance, one line per age group
print("  16b. Line graph: Avg Education by Observance × Age")
edu_numeric = {'Some High School': 1, 'High School diploma or GED': 2,
               'Associates Degree': 3, 'Bachelors Degree': 4,
               'Masters Degree': 5, 'Doctorate or PhD': 6}
df['edu_num'] = df['education'].map(edu_numeric)
fig, ax = plt.subplots(figsize=(10, 6))
for idx, age in enumerate(ages_present):
    sub = df[(df['age'] == age) & df['observance'].isin(OBS_ORDER) & df['edu_num'].notna()]
    means = sub.groupby('observance')['edu_num'].mean().reindex(OBS_ORDER).dropna()
    ax.plot(means.index, means.values, 'o-', label=f"{age} (n={len(sub)})",
            color=LINE_COLORS[idx], linewidth=2.5, markersize=8)
ax.set_title(f"Avg. Education Level by Observance, by Age ({scope_label})")
ax.set_ylabel("Avg. Education Score\n(1=<HS → 6=Doctorate)")
ax.set_xlabel("Observance Level")
ax.set_ylim(0.5, 6.5); ax.grid(True, alpha=0.3)
ax.set_yticks(range(1, 7))
ax.set_yticklabels(['<HS', 'HS/GED', 'Associates', 'Bachelors', 'Masters', 'Doctorate'])
ax.legend(title="Age Group")
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
save(fig, "16b_education_by_observance_age_lines")

# ═══════════════════════════════════════════════════════════════════════
# 17. ★ NEW: BIRTH REGION BREAKDOWN
#     (Inspired by Pew: generational / immigration origin)
# ═══════════════════════════════════════════════════════════════════════
print("\n17. Birth Region")
region_order = ['Central Asia / FSU', 'Israel', 'New York City', 'Other US', 'Europe / Other']
bar_chart(df['birth_region'], "Birth Region of Respondents", "Region",
          order=region_order, name="17_birth_region")
pie_chart(df['birth_region'], "Birth Region", name="17_birth_region_pie")

# ═══════════════════════════════════════════════════════════════════════
# 18. ★ NEW: OBSERVANCE BY BIRTH REGION
#     (Pew-style: does religiosity differ for immigrants vs US-born?)
# ═══════════════════════════════════════════════════════════════════════
print("\n18. Observance by Birth Region")
ct_br = reindex_ct(pd.crosstab(df['birth_region'], df['observance']),
                   region_order, OBS_ORDER)
ct_br_norm = ct_br.div(ct_br.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(11, 6))
ct_br_norm.plot.bar(ax=ax, stacked=True, colormap='Set2')
ax.set_title(f"Observance by Birth Region ({scope_label})")
ax.set_ylabel("Percentage"); ax.set_xlabel("Birth Region")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Observance", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
save(fig, "18_observance_by_birth_region")

# 18b. LINE: observance % across birth regions
print("  18b. Line graph: Observance by Birth Region")
fig, ax = plt.subplots(figsize=(10, 6))
for idx, obs in enumerate(OBS_ORDER):
    if obs in ct_br_norm.columns:
        ax.plot(ct_br_norm.index, ct_br_norm[obs].values, 'o-', label=obs,
                color=LINE_COLORS[idx], linewidth=2.5, markersize=8)
ax.set_title(f"Observance (%) by Birth Region ({scope_label})")
ax.set_ylabel("% of Birth Region"); ax.set_xlabel("Birth Region")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(True, alpha=0.3); ax.legend(title="Observance")
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
save(fig, "18b_observance_by_birth_region_lines")

# ═══════════════════════════════════════════════════════════════════════
# 19. ★ NEW: EDUCATION BY BIRTH REGION
#     (Immigrant attainment vs US-born — classic Queens immigrant Q)
# ═══════════════════════════════════════════════════════════════════════
print("\n19. Education by Birth Region")
ct_bre = reindex_ct(pd.crosstab(df['birth_region'], df['education']),
                    region_order, EDU_ORDER)
ct_bre_norm = ct_bre.div(ct_bre.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(11, 6))
ct_bre_norm.plot.bar(ax=ax, stacked=True, colormap='tab20')
ax.set_title(f"Education by Birth Region ({scope_label})")
ax.set_ylabel("Percentage"); ax.set_xlabel("Birth Region")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Education", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
save(fig, "19_education_by_birth_region")

# 19b. LINE: avg education score by birth region
print("  19b. Line graph: Avg Education by Birth Region")
df_br_edu = df[df['birth_region'].isin(region_order) & df['edu_num'].notna()].copy()
fig, ax = plt.subplots(figsize=(10, 6))
means_bre = df_br_edu.groupby('birth_region')['edu_num'].mean().reindex(region_order).dropna()
ax.plot(means_bre.index, means_bre.values, 'o-', color=LINE_COLORS[2], linewidth=2.5, markersize=9)
ax.set_title(f"Avg. Education Level by Birth Region ({scope_label})")
ax.set_ylabel("Avg. Education Score\n(1=<HS → 6=Doctorate)"); ax.set_xlabel("Birth Region")
ax.set_ylim(0.5, 6.5); ax.grid(True, alpha=0.3)
ax.set_yticks(range(1, 7))
ax.set_yticklabels(['<HS', 'HS/GED', 'Associates', 'Bachelors', 'Masters', 'Doctorate'])
for i, v in enumerate(means_bre.values):
    ax.text(i, v + 0.15, f"{v:.2f}", ha='center', fontsize=11)
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
save(fig, "19b_education_by_birth_region_line")

# ═══════════════════════════════════════════════════════════════════════
# 20. ★ NEW: FARSI PROFICIENCY BY BIRTH REGION
#     (Language retention: FSU-born vs US-born)
# ═══════════════════════════════════════════════════════════════════════
print("\n20. Farsi by Birth Region")
ct_brf = reindex_ct(pd.crosstab(df['birth_region'], df['farsi']),
                    region_order, FARSI_ORDER)
ct_brf_norm = ct_brf.div(ct_brf.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(11, 6))
ct_brf_norm.plot.bar(ax=ax, stacked=True, colormap='RdYlGn')
ax.set_title(f"Farsi Proficiency by Birth Region ({scope_label})")
ax.set_ylabel("Percentage"); ax.set_xlabel("Birth Region")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Farsi Level", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
save(fig, "20_farsi_by_birth_region")

# 20b. LINE: avg Farsi score by birth region
print("  20b. Line graph: Avg Farsi by Birth Region")
df_br_farsi = df[df['birth_region'].isin(region_order) & df['farsi_num'].notna()].copy()
fig, ax = plt.subplots(figsize=(10, 6))
means_brf = df_br_farsi.groupby('birth_region')['farsi_num'].mean().reindex(region_order).dropna()
ax.plot(means_brf.index, means_brf.values, 'o-', color=LINE_COLORS[0], linewidth=2.5, markersize=9)
ax.set_title(f"Avg. Farsi Proficiency by Birth Region ({scope_label})")
ax.set_ylabel("Avg. Farsi Score\n(1=None → 4=Fluent)"); ax.set_xlabel("Birth Region")
ax.set_ylim(0.8, 4.2); ax.grid(True, alpha=0.3)
for i, v in enumerate(means_brf.values):
    ax.text(i, v + 0.1, f"{v:.2f}", ha='center', fontsize=11)
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
save(fig, "20b_farsi_by_birth_region_line")

# ═══════════════════════════════════════════════════════════════════════
# 21. ★ NEW: COMMUNITY MEMBERSHIP BY AGE AND OBSERVANCE
#     (Pew-style: engagement correlates)
# ═══════════════════════════════════════════════════════════════════════
print("\n21. Community Membership by Age & Observance")
mem = df[df['is_member'].isin(['Yes', 'No'])].copy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# By age
ct_ma = pd.crosstab(mem['age'], mem['is_member'])
ct_ma = ct_ma.reindex(index=[a for a in AGE_ORDER if a in ct_ma.index])
pct_yes_age = (ct_ma['Yes'] / ct_ma.sum(axis=1) * 100)
pct_yes_age.plot.bar(ax=ax1, color='steelblue')
ax1.set_title("Membership Rate by Age"); ax1.set_ylabel("% Members")
ax1.yaxis.set_major_formatter(mticker.PercentFormatter())
ax1.set_ylim(0, 105)
for i, v in enumerate(pct_yes_age.values):
    ax1.text(i, v + 1, f"{v:.0f}%", ha='center', fontsize=10)
ax1.tick_params(axis='x', rotation=45)

# By observance
ct_mo = pd.crosstab(mem['observance'], mem['is_member'])
ct_mo = ct_mo.reindex(index=[o for o in OBS_ORDER if o in ct_mo.index])
pct_yes_obs = (ct_mo['Yes'] / ct_mo.sum(axis=1) * 100)
pct_yes_obs.plot.bar(ax=ax2, color='darkorange')
ax2.set_title("Membership Rate by Observance"); ax2.set_ylabel("% Members")
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
ax2.set_ylim(0, 105)
for i, v in enumerate(pct_yes_obs.values):
    ax2.text(i, v + 1, f"{v:.0f}%", ha='center', fontsize=10)
ax2.tick_params(axis='x', rotation=45)

fig.suptitle(f"Community Membership Rates ({scope_label})", fontsize=14)
plt.tight_layout()
save(fig, "21_membership_by_age_observance")

# ═══════════════════════════════════════════════════════════════════════
# 22. ★ NEW: NUMBER OF HOME LANGUAGES
#     (Multilingualism patterns — common in immigrant community surveys)
# ═══════════════════════════════════════════════════════════════════════
print("\n22. Number of Home Languages")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

nl = df['num_home_langs'].dropna().astype(int)
nl_counts = nl.value_counts().sort_index()
nl_total = nl_counts.sum()
nl_counts.plot.bar(ax=ax1, color=PALETTE[:5])
ax1.set_title("Distribution of Home Languages Spoken"); ax1.set_xlabel("Number of Languages")
ax1.set_ylabel("Count")
for i, v in enumerate(nl_counts.values):
    pct = v / nl_total * 100
    ax1.text(i, v + nl_counts.max() * 0.01, f"{v}\n({pct:.0f}%)", ha='center', fontsize=10)
ax1.tick_params(axis='x', rotation=0)

ct_nla = pd.crosstab(df['age'], df['num_home_langs'].fillna(0).astype(int))
ct_nla = ct_nla.reindex(index=[a for a in AGE_ORDER if a in ct_nla.index])
ct_nla_avg = (ct_nla * ct_nla.columns).sum(axis=1) / ct_nla.sum(axis=1)
ct_nla_avg.plot.bar(ax=ax2, color='teal')
ax2.set_title("Avg. Home Languages by Age Group"); ax2.set_xlabel("Age Group")
ax2.set_ylabel("Avg. Languages")
for i, v in enumerate(ct_nla_avg.values):
    ax2.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=9)
ax2.tick_params(axis='x', rotation=45)

fig.suptitle(f"Home Language Multilingualism ({scope_label})", fontsize=14)
plt.tight_layout()
save(fig, "22_num_home_languages")

# ═══════════════════════════════════════════════════════════════════════
# 23. ★ NEW: EDUCATION × FARSI PROFICIENCY
#     (Does higher education correlate with language loss?)
# ═══════════════════════════════════════════════════════════════════════
print("\n23. Education × Farsi")
ct_ef = reindex_ct(pd.crosstab(df['education'], df['farsi']), EDU_ORDER, FARSI_ORDER)
ct_ef_norm = ct_ef.div(ct_ef.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(11, 6))
ct_ef_norm.plot.bar(ax=ax, stacked=True, colormap='RdYlGn')
ax.set_title(f"Farsi Proficiency by Education Level ({scope_label})")
ax.set_ylabel("Percentage"); ax.set_xlabel("Education")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Farsi Level", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
save(fig, "23_education_x_farsi_pct")

# ═══════════════════════════════════════════════════════════════════════
# 24. ★ NEW: BIRTH REGION × AGE
#     (Generational shift: are younger respondents more likely US-born?)
# ═══════════════════════════════════════════════════════════════════════
print("\n24. Birth Region by Age")
ct_bra = reindex_ct(pd.crosstab(df['age'], df['birth_region']), AGE_ORDER, region_order)
ct_bra_norm = ct_bra.div(ct_bra.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(12, 6))
ct_bra_norm.plot.bar(ax=ax, stacked=True, colormap='Paired')
ax.set_title(f"Birth Region by Age Group — Normalized ({scope_label})")
ax.set_ylabel("Percentage"); ax.set_xlabel("Age Group")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Birth Region", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
save(fig, "24_birth_region_by_age")

# 24b. LINE: % born in each region across age groups
print("  24b. Line graph: Birth Region % by Age")
fig, ax = plt.subplots(figsize=(10, 6))
for idx, reg in enumerate(region_order):
    if reg in ct_bra_norm.columns:
        ax.plot(ct_bra_norm.index, ct_bra_norm[reg].values, 'o-', label=reg,
                color=LINE_COLORS[idx], linewidth=2.5, markersize=8)
ax.set_title(f"Birth Region (%) by Age Group ({scope_label})")
ax.set_ylabel("% of Age Group"); ax.set_xlabel("Age Group")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(True, alpha=0.3); ax.legend(title="Birth Region")
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
save(fig, "24b_birth_region_by_age_lines")

# ═══════════════════════════════════════════════════════════════════════
# 25. ★ NEW: OBSERVANCE × AGE × BIRTH REGION
#     (Are US-born respondents less observant? Pew-style generational)
# ═══════════════════════════════════════════════════════════════════════
print("\n25. Observance: US-born vs. Foreign-born by Age")
df['born_us'] = df['birth_region'].isin(['New York City', 'Other US'])
df_both = df[df['age'].isin(AGE_ORDER) & df['observance'].isin(OBS_ORDER)].copy()
obs_numeric = {'Non-observant': 1, 'Traditional': 2, 'Observant': 3, 'Very Observant': 4}
df_both['obs_num'] = df_both['observance'].map(obs_numeric)

fig, ax = plt.subplots(figsize=(10, 6))
for born, label, color in [(True, 'US-Born', LINE_COLORS[2]), (False, 'Foreign-Born', LINE_COLORS[3])]:
    sub = df_both[df_both['born_us'] == born]
    means = sub.groupby('age')['obs_num'].mean().reindex(AGE_ORDER).dropna()
    ax.plot(means.index, means.values, 'o-', label=label, color=color, linewidth=2.5, markersize=9)

ax.set_title(f"Avg. Observance Score by Age: US-Born vs Foreign-Born ({scope_label})")
ax.set_ylabel("Avg. Observance (1=Non-obs → 4=Very Obs)")
ax.set_xlabel("Age Group")
ax.legend(); ax.set_ylim(1, 4); ax.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
save(fig, "25_observance_usborn_vs_foreign")

# ═══════════════════════════════════════════════════════════════════════
# 26. ★ NEW: NYC NEIGHBORHOOD BREAKDOWN (for US scope)
#     (Queens immigrant survey style: within-city geography)
# ═══════════════════════════════════════════════════════════════════════
if SCOPE == "us":
    print("\n26. NYC Neighborhood Breakdown")
    nyc_df = df[df['region'] == 'New York']
    nyc_total = len(nyc_df)
    nyc_cities = nyc_df['city'].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    nyc_cities.plot.barh(ax=ax, color=PALETTE[:len(nyc_cities)])
    ax.set_title(f"NYC Area — Cities & Neighborhoods (Top 15)  [N={nyc_total}]")
    ax.set_xlabel("Number of Respondents")
    ax.invert_yaxis()
    for i, v in enumerate(nyc_cities.values):
        pct = v / nyc_total * 100
        ax.text(v + nyc_cities.max() * 0.01, i, f" {v} ({pct:.1f}%)", va='center', fontsize=9)
    plt.tight_layout()
    save(fig, "26_nyc_neighborhoods")

# ═══════════════════════════════════════════════════════════════════════
# 27. OCCUPATION DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════
print("\n27. Occupation Distribution")
occ = df['occupation'].dropna()
occ_counts = occ.value_counts()
occ_total = occ_counts.sum()
fig, ax = plt.subplots(figsize=(12, 9))
occ_sorted = occ_counts.reindex([o for o in OCCUPATION_ORDER if o in occ_counts.index]).dropna()
occ_sorted.plot.barh(ax=ax, color=PALETTE[:len(occ_sorted)])
ax.set_title(f"Occupation Distribution ({scope_label})  [N={occ_total}]")
ax.set_xlabel("Number of Respondents")
ax.invert_yaxis()
for i, v in enumerate(occ_sorted.values):
    pct = v / occ_total * 100
    ax.text(v + occ_sorted.max() * 0.01, i, f" {v} ({pct:.1f}%)", va='center', fontsize=10)
plt.tight_layout()
save(fig, "27_occupation_distribution")

# ═══════════════════════════════════════════════════════════════════════
# 28. OCCUPATION — HEALTHCARE vs NON-HEALTHCARE BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════
print("\n28. Healthcare vs Non-Healthcare")
df['is_healthcare'] = df['occupation'].isin(HEALTHCARE_CATS)
df_occ = df[df['occupation'].notna() & ~df['occupation'].isin(['Student', 'Retired', 'Homemaker'])].copy()
hc_counts = df_occ['is_healthcare'].value_counts()
hc_labels = {True: f"Healthcare\n(n={hc_counts.get(True,0)})",
             False: f"Non-Healthcare\n(n={hc_counts.get(False,0)})"}
fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    hc_counts.values, labels=[hc_labels[k] for k in hc_counts.index],
    autopct=lambda p: f'{p:.1f}%', colors=[LINE_COLORS[2], LINE_COLORS[3]],
    startangle=140, pctdistance=0.75)
for t in texts:
    t.set_fontsize(13)
for t in autotexts:
    t.set_fontsize(13)
ax.set_title(f"Healthcare vs Non-Healthcare Workers ({scope_label})\n(excl. students, retired, homemakers)")
plt.tight_layout()
save(fig, "28_healthcare_vs_other")

# ═══════════════════════════════════════════════════════════════════════
# 29. OCCUPATION BY AGE GROUP
# ═══════════════════════════════════════════════════════════════════════
print("\n29. Occupation by Age — Top Categories")
top_occ = occ_counts.head(12).index.tolist()
df_top = df[df['occupation'].isin(top_occ) & df['age'].isin(AGE_ORDER)].copy()
ct_oa = pd.crosstab(df_top['occupation'], df_top['age'])
ct_oa = ct_oa.reindex(columns=[a for a in AGE_ORDER if a in ct_oa.columns])
ct_oa = ct_oa.loc[ct_oa.sum(axis=1).sort_values(ascending=False).index]
ct_oa_norm = ct_oa.div(ct_oa.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(13, 7))
ct_oa_norm.plot.barh(ax=ax, stacked=True, colormap='Set2')
ax.set_title(f"Age Composition of Top Occupations ({scope_label})")
ax.set_xlabel("Percentage"); ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Age Group", bbox_to_anchor=(1.02, 1), loc='upper left')
ax.invert_yaxis()
plt.tight_layout()
save(fig, "29_occupation_by_age")

# ═══════════════════════════════════════════════════════════════════════
# 30. OCCUPATION BY OBSERVANCE
# ═══════════════════════════════════════════════════════════════════════
print("\n30. Occupation by Observance")
df_top_obs = df[df['occupation'].isin(top_occ) & df['observance'].isin(OBS_ORDER)].copy()
ct_oo = pd.crosstab(df_top_obs['occupation'], df_top_obs['observance'])
ct_oo = ct_oo.reindex(columns=[o for o in OBS_ORDER if o in ct_oo.columns])
ct_oo = ct_oo.loc[ct_oo.sum(axis=1).sort_values(ascending=False).index]
ct_oo_norm = ct_oo.div(ct_oo.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(13, 7))
ct_oo_norm.plot.barh(ax=ax, stacked=True, colormap='Set2')
ax.set_title(f"Observance Composition of Top Occupations ({scope_label})")
ax.set_xlabel("Percentage"); ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Observance", bbox_to_anchor=(1.02, 1), loc='upper left')
ax.invert_yaxis()
plt.tight_layout()
save(fig, "30_occupation_by_observance")

# ═══════════════════════════════════════════════════════════════════════
# 31. OCCUPATION BY EDUCATION
# ═══════════════════════════════════════════════════════════════════════
print("\n31. Occupation by Education")
df_top_edu = df[df['occupation'].isin(top_occ) & df['education'].isin(EDU_ORDER)].copy()
ct_oe = pd.crosstab(df_top_edu['occupation'], df_top_edu['education'])
ct_oe = ct_oe.reindex(columns=[e for e in EDU_ORDER if e in ct_oe.columns])
ct_oe = ct_oe.loc[ct_oe.sum(axis=1).sort_values(ascending=False).index]
ct_oe.columns = [EDU_SHORT.get(c, c) for c in ct_oe.columns]
ct_oe_norm = ct_oe.div(ct_oe.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(13, 7))
ct_oe_norm.plot.barh(ax=ax, stacked=True, colormap='tab20')
ax.set_title(f"Education Composition of Top Occupations ({scope_label})")
ax.set_xlabel("Percentage"); ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Education", bbox_to_anchor=(1.02, 1), loc='upper left')
ax.invert_yaxis()
plt.tight_layout()
save(fig, "31_occupation_by_education")

# ═══════════════════════════════════════════════════════════════════════
# 32. OCCUPATION BY FARSI PROFICIENCY
# ═══════════════════════════════════════════════════════════════════════
print("\n32. Occupation by Farsi Proficiency")
df_top_farsi = df[df['occupation'].isin(top_occ) & df['farsi'].isin(FARSI_ORDER)].copy()
ct_of = pd.crosstab(df_top_farsi['occupation'], df_top_farsi['farsi'])
ct_of = ct_of.reindex(columns=[f for f in FARSI_ORDER if f in ct_of.columns])
ct_of = ct_of.loc[ct_of.sum(axis=1).sort_values(ascending=False).index]
ct_of_norm = ct_of.div(ct_of.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(13, 7))
ct_of_norm.plot.barh(ax=ax, stacked=True, colormap='RdYlGn')
ax.set_title(f"Farsi Proficiency of Top Occupations ({scope_label})")
ax.set_xlabel("Percentage"); ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Farsi Level", bbox_to_anchor=(1.02, 1), loc='upper left')
ax.invert_yaxis()
plt.tight_layout()
save(fig, "32_occupation_by_farsi")

# ═══════════════════════════════════════════════════════════════════════
# 33. OCCUPATION BY BIRTH REGION
# ═══════════════════════════════════════════════════════════════════════
print("\n33. Occupation by Birth Region")
region_order_occ = ['Central Asia / FSU', 'New York City', 'Israel', 'Other US', 'Europe / Other']
df_top_br = df[df['occupation'].isin(top_occ) & df['birth_region'].isin(region_order_occ)].copy()
ct_obr = pd.crosstab(df_top_br['occupation'], df_top_br['birth_region'])
ct_obr = ct_obr.reindex(columns=[r for r in region_order_occ if r in ct_obr.columns])
ct_obr = ct_obr.loc[ct_obr.sum(axis=1).sort_values(ascending=False).index]
ct_obr_norm = ct_obr.div(ct_obr.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(13, 7))
ct_obr_norm.plot.barh(ax=ax, stacked=True, colormap='Paired')
ax.set_title(f"Birth Region of Top Occupations ({scope_label})")
ax.set_xlabel("Percentage"); ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Birth Region", bbox_to_anchor=(1.02, 1), loc='upper left')
ax.invert_yaxis()
plt.tight_layout()
save(fig, "33_occupation_by_birth_region")

# ═══════════════════════════════════════════════════════════════════════
# 34. HEALTHCARE RATE BY AGE — LINE GRAPH
# ═══════════════════════════════════════════════════════════════════════
print("\n34. Healthcare Rate by Age & Observance")
df_working = df[df['occupation'].notna() &
                ~df['occupation'].isin(['Student', 'Retired', 'Homemaker'])].copy()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

# By age
hc_by_age = df_working.groupby('age')['is_healthcare'].mean().reindex(AGE_ORDER).dropna() * 100
ax1.plot(hc_by_age.index, hc_by_age.values, 'o-', color=LINE_COLORS[0], linewidth=2.5, markersize=9)
ax1.set_title("Healthcare Worker Rate by Age")
ax1.set_ylabel("% in Healthcare"); ax1.set_xlabel("Age Group")
ax1.yaxis.set_major_formatter(mticker.PercentFormatter())
ax1.grid(True, alpha=0.3)
for i, v in enumerate(hc_by_age.values):
    ax1.text(i, v + 1.5, f"{v:.0f}%", ha='center', fontsize=10)
ax1.tick_params(axis='x', rotation=45)

# By observance
hc_by_obs = df_working.groupby('observance')['is_healthcare'].mean().reindex(OBS_ORDER).dropna() * 100
ax2.plot(hc_by_obs.index, hc_by_obs.values, 'o-', color=LINE_COLORS[2], linewidth=2.5, markersize=9)
ax2.set_title("Healthcare Worker Rate by Observance")
ax2.set_ylabel("% in Healthcare"); ax2.set_xlabel("Observance Level")
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
ax2.grid(True, alpha=0.3)
for i, v in enumerate(hc_by_obs.values):
    ax2.text(i, v + 1.5, f"{v:.0f}%", ha='center', fontsize=10)
ax2.tick_params(axis='x', rotation=45)

fig.suptitle(f"Healthcare Employment Rates ({scope_label})", fontsize=16)
plt.tight_layout()
save(fig, "34_healthcare_rate_by_age_observance")

# ═══════════════════════════════════════════════════════════════════════
# 35. OCCUPATION DISTRIBUTION — EXCLUDING STUDENTS
# ═══════════════════════════════════════════════════════════════════════
print("\n35. Occupation Distribution (excl. Students)")
df_no_stu = df[df['occupation'].notna() & (df['occupation'] != 'Student')].copy()
occ_ns = df_no_stu['occupation'].value_counts()
occ_ns_total = occ_ns.sum()
occ_ns_sorted = occ_ns.reindex([o for o in OCCUPATION_ORDER if o in occ_ns.index and o != 'Student']).dropna()
fig, ax = plt.subplots(figsize=(12, 9))
occ_ns_sorted.plot.barh(ax=ax, color=PALETTE[:len(occ_ns_sorted)])
ax.set_title(f"Occupation Distribution — excl. Students ({scope_label})  [N={occ_ns_total}]")
ax.set_xlabel("Number of Respondents")
ax.invert_yaxis()
for i, v in enumerate(occ_ns_sorted.values):
    pct = v / occ_ns_total * 100
    ax.text(v + occ_ns_sorted.max() * 0.01, i, f" {v} ({pct:.1f}%)", va='center', fontsize=10)
plt.tight_layout()
save(fig, "35_occupation_no_students")

# ═══════════════════════════════════════════════════════════════════════
# 36. YOUTH OCCUPATIONS: UNDER 20 + 20-29, WITH AND WITHOUT STUDENTS
# ═══════════════════════════════════════════════════════════════════════
print("\n36. Youth Occupations (Under 30)")
df_youth = df[df['age'].isin(['Under 20', '20-29']) & df['occupation'].notna()].copy()
df_youth_working = df_youth[~df_youth['occupation'].isin(['Student', 'Retired', 'Homemaker'])].copy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))

# Left: with students
youth_all = df_youth['occupation'].value_counts().head(12)
youth_all_total = df_youth['occupation'].value_counts().sum()
youth_all.plot.barh(ax=ax1, color=PALETTE[:len(youth_all)])
ax1.set_title(f"Under 30 — All (N={youth_all_total})")
ax1.set_xlabel("Count")
ax1.invert_yaxis()
for i, v in enumerate(youth_all.values):
    pct = v / youth_all_total * 100
    ax1.text(v + youth_all.max() * 0.01, i, f" {v} ({pct:.0f}%)", va='center', fontsize=10)

# Right: without students
youth_work = df_youth_working['occupation'].value_counts().head(12)
youth_work_total = df_youth_working['occupation'].value_counts().sum()
youth_work.plot.barh(ax=ax2, color=PALETTE[:len(youth_work)])
ax2.set_title(f"Under 30 — Workers Only (N={youth_work_total})")
ax2.set_xlabel("Count")
ax2.invert_yaxis()
for i, v in enumerate(youth_work.values):
    pct = v / youth_work_total * 100
    ax2.text(v + youth_work.max() * 0.01, i, f" {v} ({pct:.0f}%)", va='center', fontsize=10)

fig.suptitle(f"Youth Occupations: With vs Without Students ({scope_label})", fontsize=15)
plt.tight_layout()
save(fig, "36_youth_occupations")

# ═══════════════════════════════════════════════════════════════════════
# 37. OCCUPATION BY AGE — EXCLUDING STUDENTS
# ═══════════════════════════════════════════════════════════════════════
print("\n37. Occupation by Age — excl. Students")
top_occ_ns = [o for o in occ_ns.head(12).index if o != 'Student'][:12]
df_top_ns = df[df['occupation'].isin(top_occ_ns) & df['age'].isin(AGE_ORDER)].copy()
ct_oa_ns = pd.crosstab(df_top_ns['occupation'], df_top_ns['age'])
ct_oa_ns = ct_oa_ns.reindex(columns=[a for a in AGE_ORDER if a in ct_oa_ns.columns])
ct_oa_ns = ct_oa_ns.loc[ct_oa_ns.sum(axis=1).sort_values(ascending=False).index]
ct_oa_ns_norm = ct_oa_ns.div(ct_oa_ns.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(13, 7))
ct_oa_ns_norm.plot.barh(ax=ax, stacked=True, colormap='Set2')
ax.set_title(f"Age Composition of Top Occupations — excl. Students ({scope_label})")
ax.set_xlabel("Percentage"); ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Age Group", bbox_to_anchor=(1.02, 1), loc='upper left')
ax.invert_yaxis()
plt.tight_layout()
save(fig, "37_occupation_by_age_no_students")

# ═══════════════════════════════════════════════════════════════════════
# 38. YOUTH HEALTHCARE RATE — LINE BY AGE, WITH AND WITHOUT STUDENTS
# ═══════════════════════════════════════════════════════════════════════
print("\n38. Healthcare Rate: With vs Without Students")
df_has_occ = df[df['occupation'].notna() & df['age'].isin(AGE_ORDER)].copy()
df_workers = df_has_occ[~df_has_occ['occupation'].isin(['Student', 'Retired', 'Homemaker'])].copy()

fig, ax = plt.subplots(figsize=(10, 6))

# With students (as non-healthcare)
hc_all = df_has_occ.groupby('age')['is_healthcare'].mean().reindex(AGE_ORDER).dropna() * 100
ax.plot(hc_all.index, hc_all.values, 'o--', label='All respondents (students counted as non-HC)',
        color=LINE_COLORS[5], linewidth=2, markersize=7, alpha=0.7)

# Without students/retired/homemakers
hc_work = df_workers.groupby('age')['is_healthcare'].mean().reindex(AGE_ORDER).dropna() * 100
ax.plot(hc_work.index, hc_work.values, 'o-', label='Workers only (excl. students/retired/homemakers)',
        color=LINE_COLORS[0], linewidth=2.5, markersize=9)

for i, (v1, v2) in enumerate(zip(hc_all.values, hc_work.values)):
    ax.text(i, v2 + 2, f"{v2:.0f}%", ha='center', fontsize=10, color=LINE_COLORS[0])

ax.set_title(f"Healthcare Worker Rate by Age ({scope_label})")
ax.set_ylabel("% in Healthcare"); ax.set_xlabel("Age Group")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
save(fig, "38_healthcare_rate_students_comparison")

# ═══════════════════════════════════════════════════════════════════════
# 39. YOUTH DEEP DIVE: OBSERVANCE × OCCUPATION (UNDER 30, NO STUDENTS)
# ═══════════════════════════════════════════════════════════════════════
print("\n39. Youth (Under 30) Workers: Observance by Occupation")
df_yw_obs = df_youth_working[df_youth_working['observance'].isin(OBS_ORDER)].copy()
if len(df_yw_obs) > 20:
    top_youth_occ = df_yw_obs['occupation'].value_counts().head(8).index
    df_yw_top = df_yw_obs[df_yw_obs['occupation'].isin(top_youth_occ)]
    ct_yw = pd.crosstab(df_yw_top['occupation'], df_yw_top['observance'])
    ct_yw = ct_yw.reindex(columns=[o for o in OBS_ORDER if o in ct_yw.columns])
    ct_yw = ct_yw.loc[ct_yw.sum(axis=1).sort_values(ascending=False).index]
    ct_yw_norm = ct_yw.div(ct_yw.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(13, 6))
    ct_yw_norm.plot.barh(ax=ax, stacked=True, colormap='Set2')
    ax.set_title(f"Observance by Occupation — Under 30, Workers Only ({scope_label})")
    ax.set_xlabel("Percentage"); ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(title="Observance", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.invert_yaxis()
    plt.tight_layout()
    save(fig, "39_youth_workers_observance_by_occ")

# ═══════════════════════════════════════════════════════════════════════
# 44. OCCUPATION DISTRIBUTION BY OBSERVANCE — FACETED
#     Side-by-side: what jobs do Traditional vs Observant vs
#     Non-observant people actually do?
# ═══════════════════════════════════════════════════════════════════════
print("\n44. Occupation by Observance — Faceted (workers only)")
df_workers_obs = df[df['occupation'].notna() &
                    ~df['occupation'].isin(['Student', 'Retired', 'Homemaker']) &
                    df['observance'].isin(OBS_ORDER)].copy()

obs_with_data = [o for o in OBS_ORDER if (df_workers_obs['observance'] == o).sum() >= 15]
n_panels = len(obs_with_data)
fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 8), sharey=False)
if n_panels == 1:
    axes = [axes]

for idx, obs in enumerate(obs_with_data):
    ax = axes[idx]
    sub = df_workers_obs[df_workers_obs['observance'] == obs]
    top = sub['occupation'].value_counts().head(10)
    total = top.sum()
    top.plot.barh(ax=ax, color=LINE_COLORS[OBS_ORDER.index(obs)])
    ax.set_title(f"{obs}\n(n={len(sub)} workers)", fontsize=13)
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    for i, v in enumerate(top.values):
        pct = v / len(sub) * 100
        ax.text(v + top.max() * 0.02, i, f" {v} ({pct:.0f}%)", va='center', fontsize=10)

fig.suptitle(f"Top 10 Occupations by Observance Level — Workers Only ({scope_label})",
             fontsize=15, y=1.01)
plt.tight_layout()
save(fig, "44_occupation_by_observance_faceted")

# ═══════════════════════════════════════════════════════════════════════
# 45. OCCUPATION CATEGORY RATE BY OBSERVANCE — LINE GRAPHS
#     For key occupation categories, what % of each observance level
#     works in that field?
# ═══════════════════════════════════════════════════════════════════════
print("\n45. Occupation Category Rate by Observance — Line Graphs")
key_categories = [
    ('Healthcare (all)', HEALTHCARE_CATS),
    ('Physician', ['Physician']),
    ('Nurse / NP', ['Nurse / NP']),
    ('Pharmacist', ['Pharmacist']),
    ('Dentist / Dental', ['Dentist / Dental']),
    ('Barber / Beauty', ['Barber / Beauty']),
    ('Business Owner', ['Business Owner']),
    ('Accounting / CPA', ['Accounting / CPA']),
    ('Law / Legal', ['Law / Legal']),
    ('Education', ['Education']),
    ('Tech / IT', ['Tech / IT']),
    ('Real Estate', ['Real Estate']),
]

fig, axes = plt.subplots(3, 4, figsize=(13, 10))
axes = axes.flatten()

for idx, (cat_name, cat_list) in enumerate(key_categories):
    ax = axes[idx]
    rates = []
    for obs in obs_with_data:
        sub = df_workers_obs[df_workers_obs['observance'] == obs]
        rate = sub['occupation'].isin(cat_list).mean() * 100
        rates.append(rate)
    ax.bar(range(len(obs_with_data)), rates,
           color=[LINE_COLORS[OBS_ORDER.index(o)] for o in obs_with_data])
    ax.set_xticks(range(len(obs_with_data)))
    ax.set_xticklabels([o.replace('-', '-\n') for o in obs_with_data], fontsize=8)
    ax.set_title(cat_name, fontsize=11)
    ax.set_ylabel("%" if idx % 4 == 0 else "")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    for i, v in enumerate(rates):
        ax.text(i, v + 0.5, f"{v:.0f}%", ha='center', fontsize=9)

fig.suptitle(f"% of Each Observance Group in Key Occupations — Workers Only ({scope_label})",
             fontsize=14, y=1.01)
plt.tight_layout()
save(fig, "45_occupation_rate_by_observance")

# ═══════════════════════════════════════════════════════════════════════
# 46. OBSERVANCE COMPOSITION WITHIN EACH OCCUPATION
#     Flip the question: for each occupation, what's the observance mix?
#     Compared to the overall population mix as a reference line.
# ═══════════════════════════════════════════════════════════════════════
print("\n46. Observance Mix Within Each Occupation")
# Overall observance distribution (workers only) as reference
overall_obs_dist = df_workers_obs['observance'].value_counts(normalize=True).reindex(OBS_ORDER).fillna(0) * 100

top_occ_for_obs = df_workers_obs['occupation'].value_counts().head(14).index
df_occ_obs = df_workers_obs[df_workers_obs['occupation'].isin(top_occ_for_obs)]
ct_occ_obs = pd.crosstab(df_occ_obs['occupation'], df_occ_obs['observance'])
ct_occ_obs = ct_occ_obs.reindex(columns=[o for o in OBS_ORDER if o in ct_occ_obs.columns])
ct_occ_obs = ct_occ_obs.loc[ct_occ_obs.sum(axis=1).sort_values(ascending=False).index]

# Append population average as a reference row
pop_row = pd.DataFrame([overall_obs_dist.reindex(ct_occ_obs.columns).values],
                       columns=ct_occ_obs.columns, index=['▸ POPULATION AVG'])
ct_occ_obs_with_ref = pd.concat([pop_row, ct_occ_obs])
ct_occ_obs_norm = ct_occ_obs_with_ref.div(ct_occ_obs_with_ref.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(13, 8))
ct_occ_obs_norm.plot.barh(ax=ax, stacked=True, colormap='Set2')
ax.set_title(f"Observance Composition Within Each Occupation ({scope_label})\n"
             f"Top row = overall population average for reference")
ax.set_xlabel("Percentage"); ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Observance", bbox_to_anchor=(1.02, 1), loc='upper left')
ax.invert_yaxis()

# Bold the reference row label
labels = ax.get_yticklabels()
for lbl in labels:
    if 'POPULATION' in lbl.get_text():
        lbl.set_fontweight('bold')
        lbl.set_color('red')

plt.tight_layout()
save(fig, "46_observance_within_occupation")

# ═══════════════════════════════════════════════════════════════════════
# 47. HEALTHCARE SUBCATEGORY BREAKDOWN BY OBSERVANCE
#     Within healthcare, do observance levels differ for physicians
#     vs nurses vs pharmacists etc?
# ═══════════════════════════════════════════════════════════════════════
print("\n47. Healthcare Subcategories by Observance")
df_hc = df_workers_obs[df_workers_obs['occupation'].isin(HEALTHCARE_CATS)].copy()
hc_cats_present = [c for c in HEALTHCARE_CATS
                   if (df_hc['occupation'] == c).sum() >= 5]

if len(hc_cats_present) >= 3:
    ct_hc_obs = pd.crosstab(df_hc['occupation'], df_hc['observance'])
    ct_hc_obs = ct_hc_obs.reindex(
        index=[c for c in hc_cats_present if c in ct_hc_obs.index],
        columns=[o for o in OBS_ORDER if o in ct_hc_obs.columns])
    ct_hc_obs = ct_hc_obs.loc[ct_hc_obs.sum(axis=1).sort_values(ascending=False).index]
    ct_hc_obs_norm = ct_hc_obs.div(ct_hc_obs.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    ct_hc_obs_norm.plot.barh(ax=ax, stacked=True, colormap='Set2')
    ax.set_title(f"Observance Within Healthcare Subcategories ({scope_label})")
    ax.set_xlabel("Percentage"); ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(title="Observance", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.invert_yaxis()
    # Add n= labels
    for i, cat in enumerate(ct_hc_obs.index):
        n = ct_hc_obs.loc[cat].sum()
        ax.text(101, i, f" n={int(n)}", va='center', fontsize=10)
    plt.tight_layout()
    save(fig, "47_healthcare_subcategories_by_observance")

# ═══════════════════════════════════════════════════════════════════════
# 40–43. DEVIATION FROM BASELINE — BY AGE GROUP
#   For each age group, compute baselines for key metrics, then show
#   how subgroups (by observance, birth region) deviate from that average.
# ═══════════════════════════════════════════════════════════════════════

# Pre-compute numeric columns (some may already exist from earlier)
if 'farsi_num' not in df.columns:
    farsi_numeric = {"Don't know it at all": 1, 'Only know food names and swear words': 2,
                     'Limited Proficiency': 3, 'Fluent': 4}
    df['farsi_num'] = df['farsi'].map(farsi_numeric)
if 'edu_num' not in df.columns:
    edu_numeric = {'Some High School': 1, 'High School diploma or GED': 2,
                   'Associates Degree': 3, 'Bachelors Degree': 4,
                   'Masters Degree': 5, 'Doctorate or PhD': 6}
    df['edu_num'] = df['education'].map(edu_numeric)
if 'obs_num' not in df.columns:
    obs_numeric_map = {'Non-observant': 1, 'Traditional': 2, 'Observant': 3, 'Very Observant': 4}
    df['obs_num'] = df['observance'].map(obs_numeric_map)
if 'is_healthcare' not in df.columns:
    df['is_healthcare'] = df['occupation'].isin(HEALTHCARE_CATS)

METRIC_DEFS = [
    ('farsi_num', 'Farsi Proficiency\n(1–4 scale)', 'score'),
    ('edu_num', 'Education Level\n(1–6 scale)', 'score'),
    ('obs_num', 'Observance Level\n(1–4 scale)', 'score'),
    ('hc_pct', '% in Healthcare', 'pct'),
    ('nyc_born_pct', '% Born in NYC', 'pct'),
]

def compute_profile(sub):
    """Compute metric averages for a subgroup."""
    return {
        'farsi_num': sub['farsi_num'].mean(),
        'edu_num': sub['edu_num'].mean(),
        'obs_num': sub['obs_num'].mean(),
        'hc_pct': sub['is_healthcare'].mean() * 100 if sub['is_healthcare'].notna().any() else np.nan,
        'nyc_born_pct': sub['birth_region'].eq('New York City').mean() * 100 if sub['birth_region'].notna().any() else np.nan,
    }

def deviation_chart(ax, baseline, subgroups, metrics, colors, label_suffix=""):
    """Draw a diverging bar chart of absolute deviations from baseline."""
    y_pos = np.arange(len(metrics))
    n = len(subgroups)
    bar_h = 0.75 / max(n, 1)
    for j, (name, prof, n_grp) in enumerate(subgroups):
        vals = []
        for m, _, mtype in metrics:
            b = baseline[m]
            p = prof[m]
            if pd.notna(b) and pd.notna(p):
                vals.append(p - b)
            else:
                vals.append(0)
        ax.barh(y_pos + j * bar_h - 0.375 + bar_h / 2, vals, bar_h * 0.9,
                label=f"{name} (n={n_grp})", color=colors[j % len(colors)], alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for _, lbl, _ in metrics], fontsize=10)
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(True, axis='x', alpha=0.2)

# Metrics for observance-based charts (exclude obs_num since it's the grouping variable)
METRICS_BY_OBS = [m for m in METRIC_DEFS if m[0] != 'obs_num']
# Metrics for birth-region charts (exclude nyc_born_pct since it's tautological)
METRICS_BY_BR = [m for m in METRIC_DEFS if m[0] != 'nyc_born_pct']
# All metrics
METRICS_ALL = list(METRIC_DEFS)

# ── 40. Deviation by Observance within each age group ──────────────────
print("\n40. Deviation from Baseline by Observance (per age group)")
fig, axes = plt.subplots(2, 3, figsize=(13, 9))
axes = axes.flatten()
ages_for_dev = [a for a in AGE_ORDER if a in df['age'].values]

for idx, age in enumerate(ages_for_dev):
    ax = axes[idx]
    sub = df[df['age'] == age]
    baseline = compute_profile(sub)
    subgroups = []
    for obs in OBS_ORDER:
        grp = sub[sub['observance'] == obs]
        if len(grp) >= 5:
            subgroups.append((obs, compute_profile(grp), len(grp)))
    obs_colors = [LINE_COLORS[OBS_ORDER.index(o)] for o, _, _ in subgroups]
    deviation_chart(ax, baseline, subgroups, METRICS_BY_OBS, obs_colors)
    ax.set_title(f"Age {age} (n={len(sub)})", fontsize=12)
    ax.set_xlabel("Absolute difference from avg" if idx >= 3 else "")

for idx in range(len(ages_for_dev), len(axes)):
    axes[idx].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Observance", loc='lower right',
           bbox_to_anchor=(0.98, 0.02), fontsize=9)
fig.suptitle(f"How Observance Subgroups Deviate from Their Age-Group Average ({scope_label})",
             fontsize=14, y=1.01)
plt.tight_layout()
save(fig, "40_deviation_by_observance")

# ── 41. Deviation by Birth Region within each age group ────────────────
print("\n41. Deviation from Baseline by Birth Region (per age group)")
birth_groups = ['Central Asia / FSU', 'New York City']
fig, axes = plt.subplots(2, 3, figsize=(13, 9))
axes = axes.flatten()

for idx, age in enumerate(ages_for_dev):
    ax = axes[idx]
    sub = df[df['age'] == age]
    baseline = compute_profile(sub)
    subgroups = []
    for br in birth_groups:
        grp = sub[sub['birth_region'] == br]
        if len(grp) >= 5:
            subgroups.append((br, compute_profile(grp), len(grp)))
    br_colors = [LINE_COLORS[4 + birth_groups.index(b)] for b, _, _ in subgroups]
    deviation_chart(ax, baseline, subgroups, METRICS_BY_BR, br_colors)
    ax.set_title(f"Age {age} (n={len(sub)})", fontsize=12)
    ax.set_xlabel("Absolute difference from avg" if idx >= 3 else "")

for idx in range(len(ages_for_dev), len(axes)):
    axes[idx].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Birth Region", loc='lower right',
           bbox_to_anchor=(0.98, 0.02), fontsize=9)
fig.suptitle(f"How Birth Region Subgroups Deviate from Their Age-Group Average ({scope_label})",
             fontsize=14, y=1.01)
plt.tight_layout()
save(fig, "41_deviation_by_birth_region")

# ── 42. Single-panel summary: 30-39 deep dive ─────────────────────────
print("\n42. Baseline Profile: 30-39 Deep Dive")
ref_age = '30-39'
sub_ref = df[df['age'] == ref_age]
baseline_ref = compute_profile(sub_ref)

subgroups_42 = []
for obs in OBS_ORDER:
    grp = sub_ref[sub_ref['observance'] == obs]
    if len(grp) >= 5:
        subgroups_42.append((obs, compute_profile(grp), len(grp)))
for br in ['Central Asia / FSU', 'New York City']:
    grp = sub_ref[sub_ref['birth_region'] == br]
    if len(grp) >= 5:
        subgroups_42.append((f"Born: {br}", compute_profile(grp), len(grp)))

# Use all metrics except nyc_born_pct (tautological for birth-region groups)
metrics_42 = [m for m in METRIC_DEFS if m[0] != 'nyc_born_pct']
fig, ax = plt.subplots(figsize=(12, 7))
deviation_chart(ax, baseline_ref, subgroups_42, metrics_42, LINE_COLORS)
ax.set_xlabel("Absolute Difference from 30-39 Average", fontsize=12)
ax.set_title(f"30-39 Age Group: How Subgroups Deviate from Baseline ({scope_label})\n"
             f"0 = avg 30-39 respondent (n={len(sub_ref)})", fontsize=14)
ax.legend(loc='best', fontsize=9)
plt.tight_layout()
save(fig, "42_baseline_30_39_deep_dive")

# ── 43. Multi-age baseline comparison ──────────────────────────────────
print("\n43. Multi-Age Baseline: How Each Age Deviates from Overall")
overall = compute_profile(df)
subgroups_43 = []
for age in ages_for_dev:
    sub = df[df['age'] == age]
    subgroups_43.append((age, compute_profile(sub), len(sub)))

fig, ax = plt.subplots(figsize=(12, 7))
deviation_chart(ax, overall, subgroups_43, METRICS_ALL, LINE_COLORS)
ax.set_xlabel("Absolute Difference from Overall Average", fontsize=12)
ax.set_title(f"How Each Age Group Deviates from the Overall Average ({scope_label})", fontsize=14)
ax.legend(title="Age Group", loc='best', fontsize=9)
plt.tight_layout()
save(fig, "43_age_deviation_from_overall")

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"SUMMARY STATISTICS — {scope_label}")
print("=" * 60)
for col in ['age', 'education', 'observance', 'farsi', 'is_member', 'birth_region', 'occupation']:
    print(f"\n--- {col.upper()} ---")
    print(df[col].value_counts().to_string())
    print(f"  (missing: {df[col].isna().sum()})")

total = len(list(OUTPUT_DIR.glob('*.png')))
print(f"\n{'=' * 60}")
print(f"All {total} charts saved to: {OUTPUT_DIR}")
