"""Model attributes the listings don't contain: body type, seats and size class.

Curated by hand from manufacturers' standard configurations for the models in the dataset. Body type and size are
the *main* version of each model (a Golf is listed as a hatchback even though an estate exists). Treat these as
guidance and check the advert. Models not listed here are kept, with body "unknown".
Size: city < small < medium < large < xlarge.
"""
from __future__ import annotations

# (body, seats, size)
H5, S5, E5, U5, U7, M5, M7, C4, V4 = ("hatchback", 5), ("saloon", 5), ("estate", 5), ("SUV", 5), ("SUV", 7), ("MPV", 5), ("MPV", 7), ("coupe", 4), ("convertible", 4)

CATALOGUE: dict[str, tuple[str, int, str]] = {
    # Ford
    "Ford Ka+": (*H5, "city"), "Ford KA": (*H5, "city"), "Ford Streetka": ("convertible", 2, "city"), "Ford Fiesta": (*H5, "small"),
    "Ford Fusion": (*H5, "small"), "Ford Puma": (*U5, "small"), "Ford EcoSport": (*U5, "small"), "Ford B-MAX": (*M5, "small"),
    "Ford Focus": (*H5, "medium"), "Ford C-MAX": (*M5, "medium"), "Ford Grand C-MAX": (*M7, "medium"), "Ford Kuga": (*U5, "medium"),
    "Ford Mondeo": (*S5, "large"), "Ford S-MAX": (*M7, "large"), "Ford Galaxy": (*M7, "large"), "Ford Edge": (*U5, "large"),
    "Ford Mustang": (*C4, "large"), "Ford Tourneo Connect": (*M5, "medium"), "Ford Grand Tourneo Connect": (*M7, "medium"),
    "Ford Tourneo Custom": ("MPV", 8, "xlarge"), "Ford Transit Tourneo": ("MPV", 9, "xlarge"), "Ford Ranger": ("pickup", 5, "xlarge"),
    "Ford Escort": (*H5, "medium"),
    # Volkswagen
    "Volkswagen Up": (*H5, "city"), "Volkswagen Fox": (*H5, "city"), "Volkswagen Polo": (*H5, "small"), "Volkswagen T-Cross": (*U5, "small"),
    "Volkswagen Beetle": ("hatchback", 4, "small"), "Volkswagen Golf": (*H5, "medium"), "Volkswagen Golf SV": (*M5, "medium"),
    "Volkswagen Jetta": (*S5, "medium"), "Volkswagen Scirocco": (*C4, "medium"), "Volkswagen T-Roc": (*U5, "medium"),
    "Volkswagen Touran": (*M7, "medium"), "Volkswagen Tiguan": (*U5, "medium"), "Volkswagen Tiguan Allspace": (*U7, "large"),
    "Volkswagen Passat": (*S5, "large"), "Volkswagen CC": ("saloon", 4, "large"), "Volkswagen Arteon": (*S5, "large"),
    "Volkswagen Sharan": (*M7, "large"), "Volkswagen Touareg": (*U5, "xlarge"), "Volkswagen Eos": (*V4, "medium"),
    "Volkswagen Caddy Life": (*M5, "medium"), "Volkswagen Caddy Maxi Life": (*M7, "large"), "Volkswagen Caddy": (*M5, "medium"),
    "Volkswagen Caddy Maxi": (*M7, "large"), "Volkswagen Caravelle": ("MPV", 8, "xlarge"), "Volkswagen Shuttle": ("MPV", 9, "xlarge"),
    "Volkswagen California": ("campervan", 4, "xlarge"), "Volkswagen Amarok": ("pickup", 5, "xlarge"),
    # Vauxhall
    "Vauxhall Viva": (*H5, "city"), "Vauxhall Agila": (*H5, "city"), "Vauxhall Adam": ("hatchback", 4, "city"), "Vauxhall Corsa": (*H5, "small"),
    "Vauxhall Tigra": ("convertible", 2, "small"), "Vauxhall Meriva": (*M5, "small"), "Vauxhall Crossland X": (*U5, "small"),
    "Vauxhall Mokka": (*U5, "small"), "Vauxhall Mokka X": (*U5, "small"), "Vauxhall Astra": (*H5, "medium"), "Vauxhall GTC": ("hatchback", 4, "medium"),
    "Vauxhall Cascada": (*V4, "medium"), "Vauxhall Zafira": (*M7, "medium"), "Vauxhall Zafira Tourer": (*M7, "large"),
    "Vauxhall Combo Life": (*M5, "medium"), "Vauxhall Grandland X": (*U5, "medium"), "Vauxhall Insignia": (*H5, "large"),
    "Vauxhall Vectra": (*S5, "large"), "Vauxhall Antara": (*U5, "large"), "Vauxhall Ampera": (*H5, "medium"), "Vauxhall Vivaro": ("MPV", 9, "xlarge"),
    # BMW
    "BMW 1 Series": (*H5, "medium"), "BMW 2 Series": ("coupe", 4, "medium"), "BMW 3 Series": (*S5, "large"), "BMW 4 Series": (*C4, "large"),
    "BMW 5 Series": (*S5, "xlarge"), "BMW 6 Series": (*C4, "xlarge"), "BMW 7 Series": (*S5, "xlarge"), "BMW 8 Series": (*C4, "xlarge"),
    "BMW X1": (*U5, "medium"), "BMW X2": (*U5, "medium"), "BMW X3": (*U5, "large"), "BMW X4": (*U5, "large"), "BMW X5": (*U5, "xlarge"),
    "BMW X6": (*U5, "xlarge"), "BMW X7": (*U7, "xlarge"), "BMW Z3": ("convertible", 2, "small"), "BMW Z4": ("convertible", 2, "medium"),
    "BMW M2": (*C4, "medium"), "BMW M3": (*S5, "large"), "BMW M4": (*C4, "large"), "BMW M5": (*S5, "xlarge"), "BMW M6": (*C4, "xlarge"),
    "BMW i3": ("hatchback", 4, "small"), "BMW i8": ("coupe", 4, "large"),
    # Audi
    "Audi A1": (*H5, "small"), "Audi A2": (*H5, "small"), "Audi A3": (*H5, "medium"), "Audi S3": (*H5, "medium"), "Audi RS3": (*H5, "medium"),
    "Audi A4": (*S5, "large"), "Audi S4": (*S5, "large"), "Audi RS4": (*E5, "large"), "Audi A5": (*C4, "large"), "Audi S5": (*C4, "large"),
    "Audi RS5": (*C4, "large"), "Audi A6": (*S5, "xlarge"), "Audi RS6": (*E5, "xlarge"), "Audi A7": (*H5, "xlarge"), "Audi S8": (*S5, "xlarge"),
    "Audi A8": (*S5, "xlarge"), "Audi Q2": (*U5, "small"), "Audi Q3": (*U5, "medium"), "Audi Q5": (*U5, "large"), "Audi SQ5": (*U5, "large"),
    "Audi Q7": (*U7, "xlarge"), "Audi SQ7": (*U7, "xlarge"), "Audi Q8": (*U5, "xlarge"), "Audi TT": ("coupe", 4, "medium"),
    "Audi R8": ("coupe", 2, "large"), "Audi RS7": (*H5, "xlarge"),
    # Mercedes-Benz
    "Mercedes-Benz A Class": (*H5, "medium"), "Mercedes-Benz B Class": (*M5, "medium"), "Mercedes-Benz C Class": (*S5, "large"),
    "Mercedes-Benz CLA Class": (*S5, "medium"), "Mercedes-Benz CLC Class": (*C4, "medium"), "Mercedes-Benz CLK": (*C4, "large"),
    "Mercedes-Benz E Class": (*S5, "xlarge"), "Mercedes-Benz CLS Class": (*S5, "xlarge"), "Mercedes-Benz S Class": (*S5, "xlarge"),
    "Mercedes-Benz CL Class": (*C4, "xlarge"), "Mercedes-Benz GLA Class": (*U5, "medium"), "Mercedes-Benz GLB Class": (*U7, "medium"),
    "Mercedes-Benz GLC Class": (*U5, "large"), "Mercedes-Benz GLE Class": (*U5, "xlarge"), "Mercedes-Benz M Class": (*U5, "xlarge"),
    "Mercedes-Benz GL Class": (*U7, "xlarge"), "Mercedes-Benz GLS Class": (*U7, "xlarge"), "Mercedes-Benz G Class": (*U5, "xlarge"),
    "Mercedes-Benz SL CLASS": ("convertible", 2, "large"), "Mercedes-Benz SLK": ("convertible", 2, "medium"),
    "Mercedes-Benz V Class": ("MPV", 7, "xlarge"), "Mercedes-Benz X-CLASS": ("pickup", 5, "xlarge"), "Mercedes-Benz R Class": (*M7, "xlarge"),
    # Toyota
    "Toyota Aygo": ("hatchback", 4, "city"), "Toyota IQ": ("hatchback", 4, "city"), "Toyota Yaris": (*H5, "small"),
    "Toyota Urban Cruiser": (*U5, "small"), "Toyota Verso-S": (*M5, "small"), "Toyota Auris": (*H5, "medium"), "Toyota Corolla": (*H5, "medium"),
    "Toyota Prius": (*H5, "medium"), "Toyota C-HR": (*U5, "medium"), "Toyota Verso": (*M7, "medium"), "Toyota Avensis": (*S5, "large"),
    "Toyota Camry": (*S5, "large"), "Toyota RAV4": (*U5, "large"), "Toyota GT86": (*C4, "medium"), "Toyota Supra": ("coupe", 2, "medium"),
    "Toyota Land Cruiser": (*U7, "xlarge"), "Toyota Hilux": ("pickup", 5, "xlarge"), "Toyota PROACE VERSO": ("MPV", 8, "xlarge"),
    # Skoda
    "Skoda Citigo": (*H5, "city"), "Skoda Fabia": (*H5, "small"), "Skoda Kamiq": (*U5, "small"), "Skoda Roomster": (*M5, "small"),
    "Skoda Rapid": (*H5, "medium"), "Skoda Scala": (*H5, "medium"), "Skoda Octavia": (*H5, "medium"), "Skoda Karoq": (*U5, "medium"),
    "Skoda Yeti": (*U5, "medium"), "Skoda Yeti Outdoor": (*U5, "medium"), "Skoda Superb": (*H5, "large"), "Skoda Kodiaq": (*U7, "large"),
    # Hyundai
    "Hyundai i10": (*H5, "city"), "Hyundai Getz": (*H5, "small"), "Hyundai i20": (*H5, "small"), "Hyundai Accent": (*H5, "small"),
    "Hyundai I20": (*H5, "small"), "Hyundai IX20": (*M5, "small"), "Hyundai Kona": (*U5, "small"), "Hyundai i30": (*H5, "medium"),
    "Hyundai I30": (*H5, "medium"), "Hyundai Ioniq": (*H5, "medium"), "Hyundai Veloster": (*C4, "medium"), "Hyundai Tucson": (*U5, "medium"),
    "Hyundai IX35": (*U5, "medium"), "Hyundai I40": (*S5, "large"), "Hyundai i40": (*S5, "large"), "Hyundai Santa Fe": (*U7, "large"),
    "Hyundai Terracan": (*U7, "xlarge"), "Hyundai I800": ("MPV", 8, "xlarge"), "Hyundai I10": (*H5, "city"),
}
# Labels in the scrape that mix different cars, so body and seats can't be trusted: never recommended (still used for pricing).
AMBIGUOUS = {
    "Mercedes-Benz CL Class": "mostly CLA saloons (1.3-2.2L, up to 2020) though the CL coupe ended in 2014",
    "Mercedes-Benz GL Class": "mix of 7-seat GL (3.0L) and small GLA (1.6-2.1L) adverts",
}
for _k in AMBIGUOUS:
    CATALOGUE.pop(_k, None)
SIZE_ORDER = {"city": 0, "small": 1, "medium": 2, "large": 3, "xlarge": 4}


def attributes(model_key: str) -> dict:
    body, seats, size = CATALOGUE.get(model_key, ("unknown", None, "unknown"))
    return {"body": body, "seats": seats, "size": size}


def coverage(model_keys) -> float:
    keys = list(model_keys)
    return sum(k in CATALOGUE for k in keys) / max(1, len(keys))
