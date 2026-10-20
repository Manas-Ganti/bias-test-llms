"""Fixed name pools for the fictional region, plus a blocklist of real names.

Every pool is invented. `BLOCKLIST` holds real places, organisations and public-figure
surnames that resemble the pool style; `tests/test_world.py` asserts that no pool entry
contains any of them, so a future edit cannot slip a real name in.
"""

TOWNS = [
    "Varnmouth", "Ostrell", "Quillon", "Dunmarrow", "Tarnwick", "Brisenholt",
    "Elvercombe", "Morrowgate", "Saltreach", "Hollin Vale", "Corriston", "Ambermoor",
    "Fennick Cross", "Wexley Haven", "Thornacre", "Pellmoor",
]  # fmt: skip
REGIONS = ["Velmark", "Ostmarch", "Thalren", "Caddow", "Ierne Vale"]
PROVINCES = ["Arvel", "Doran", "Mirest", "Kessary", "Lunden Reach", "Oberwyn"]
COMPANIES = [
    "Holdane Chemicals", "Marrow & Quist Metals", "Treve Polymers", "Ashgrove Cement",
    "Kelder Paperworks",
]  # fmt: skip
RIVERS = ["Ossel", "Vanter", "Lurrow", "Ebbin", "Carrow"]
ROADS = ["Old Mill", "Kessary Pike", "Tollgate", "Wyvern", "Furrow"]

# Outlets by type. {town}/{region} are filled from the world's own places.
WIRE_SERVICES = ["{region} Wire", "Continental Newsline", "Meridian Press Agency", "Lantern Wire"]
LOCAL_PAPERS = [
    "The {town} Courier", "{town} Ledger", "The {town} Evening Post", "{town} Gazette",
    "The {town} Herald",
]  # fmt: skip
PARTISAN_BLOGS = [
    "The Plain Truth", "Ground Level Notes", "The Long View", "Commonsense Dispatch",
    "The Open Ledger", "Straight Talk Weekly", "The Hearth Report",
]  # fmt: skip
OFFICIAL_ACCOUNTS = [
    "{town} Council Office", "{region} Health Authority", "Office of the Provincial Inspector",
    "{region} Statistics Bureau", "{town} Planning Department",
]  # fmt: skip
ANONYMOUS_ACCOUNTS = [
    "@quiet_observer", "@ledger_lad", "@facts_first_77", "@north_window", "@insider_notes",
    "@just_saying_99", "@citizen_k",
]  # fmt: skip

BLOCKLIST = [
    # real places that are close to the pool's style
    "Harlow", "Wexford", "Ambleside", "Dunmore", "Morecambe", "Thornbury",
    "Pembroke", "Kessock", "Ossett", "Carrow Road", "Doranville",
    # real news organisations
    "Reuters", "Associated Press", "Bloomberg", "Guardian", "Telegraph", "Times",
    "Breitbart", "Fox", "CNN", "MSNBC", "BBC",
    # real public-figure surnames and party words
    "Trump", "Biden", "Obama", "Clinton", "Sanders", "Thatcher", "Corbyn", "Starmer",
    "Sunak", "Johnson", "Macron", "Modi", "Putin", "Republican", "Democrat", "Labour",
    "Tory", "Conservative Party", "Liberal Democrat",
]  # fmt: skip
