"""Country-level Google SERP markets and conventional language defaults.

Location codes verified against DataForSEO's public Google locations CSV:
https://cdn.dataforseo.com/v3/locations/locations_serp_google_2026_09_01.csv
Choose an explicit language for multilingual markets (e.g. Canada, Belgium,
Switzerland). These defaults are conveniences, not language restrictions.
"""

COUNTRY_LOCATION: dict[str, int] = {
    "al": 2008,  # Albania
    "ar": 2032,  # Argentina
    "at": 2040,  # Austria
    "au": 2036,  # Australia
    "ba": 2070,  # Bosnia and Herzegovina
    "be": 2056,  # Belgium
    "bg": 2100,  # Bulgaria
    "bo": 2068,  # Bolivia
    "br": 2076,  # Brazil
    "ca": 2124,  # Canada
    "ch": 2756,  # Switzerland
    "cl": 2152,  # Chile
    "co": 2170,  # Colombia
    "cr": 2188,  # Costa Rica
    "cy": 2196,  # Cyprus
    "cz": 2203,  # Czechia
    "de": 2276,  # Germany
    "dk": 2208,  # Denmark
    "do": 2214,  # Dominican Republic
    "ec": 2218,  # Ecuador
    "ee": 2233,  # Estonia
    "es": 2724,  # Spain
    "fi": 2246,  # Finland
    "fr": 2250,  # France
    "gb": 2826,  # United Kingdom
    "gr": 2300,  # Greece
    "gt": 2320,  # Guatemala
    "hn": 2340,  # Honduras
    "hr": 2191,  # Croatia
    "hu": 2348,  # Hungary
    "ie": 2372,  # Ireland
    "in": 2356,  # India
    "is": 2352,  # Iceland
    "it": 2380,  # Italy
    "li": 2438,  # Liechtenstein
    "lt": 2440,  # Lithuania
    "lu": 2442,  # Luxembourg
    "lv": 2428,  # Latvia
    "md": 2498,  # Moldova
    "me": 2499,  # Montenegro
    "mk": 2807,  # North Macedonia
    "mt": 2470,  # Malta
    "mx": 2484,  # Mexico
    "ni": 2558,  # Nicaragua
    "nl": 2528,  # Netherlands
    "no": 2578,  # Norway
    "nz": 2554,  # New Zealand
    "pa": 2591,  # Panama
    "pe": 2604,  # Peru
    "pl": 2616,  # Poland
    "pt": 2620,  # Portugal
    "py": 2600,  # Paraguay
    "ro": 2642,  # Romania
    "rs": 2688,  # Serbia
    "se": 2752,  # Sweden
    "si": 2705,  # Slovenia
    "sk": 2703,  # Slovakia
    "sv": 2222,  # El Salvador
    "tr": 2792,  # Turkiye
    "ua": 2804,  # Ukraine
    "us": 2840,  # United States
    "uy": 2858,  # Uruguay
    "ve": 2862,  # Venezuela
    "za": 2710,  # South Africa
}

COUNTRY_LANGUAGE: dict[str, str] = {
    "al": "sq",
    "ar": "es",
    "at": "de",
    "au": "en",
    "ba": "sr",
    "be": "nl",
    "bg": "bg",
    "bo": "es",
    "br": "pt",
    "ca": "en",
    "ch": "de",
    "cl": "es",
    "co": "es",
    "cr": "es",
    "cy": "el",
    "cz": "cs",
    "de": "de",
    "dk": "da",
    "do": "es",
    "ec": "es",
    "ee": "et",
    "es": "es",
    "fi": "fi",
    "fr": "fr",
    "gb": "en",
    "gr": "el",
    "gt": "es",
    "hn": "es",
    "hr": "hr",
    "hu": "hu",
    "ie": "en",
    "in": "en",
    "is": "is",
    "it": "it",
    "li": "de",
    "lt": "lt",
    "lu": "fr",
    "lv": "lv",
    "md": "ro",
    "me": "sr",
    "mk": "mk",
    "mt": "en",
    "mx": "es",
    "ni": "es",
    "nl": "nl",
    "no": "nb",
    "nz": "en",
    "pa": "es",
    "pe": "es",
    "pl": "pl",
    "pt": "pt",
    "py": "es",
    "ro": "ro",
    "rs": "sr",
    "se": "sv",
    "si": "sl",
    "sk": "sk",
    "sv": "es",
    "tr": "tr",
    "ua": "uk",
    "us": "en",
    "uy": "es",
    "ve": "es",
    "za": "en",
}
