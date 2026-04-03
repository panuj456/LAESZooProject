"""
GBIF Dataset Builder
Supplements the existing iNaturalist dataset with images from GBIF occurrence records.
Images are saved into the same ./data/<Species_name>/ folders used by the notebook.
"""

import requests
import os
import time
from tqdm import tqdm

SPECIES = [
    "Leptodactylus pentadactylus",      # Smoky Jungle Frog
    "Eudocimus ruber",                  # Scarlet Ibis
    "Coendou prehensilis",              # Prehensile-Tailed Porcupine
    "Uroplatus henkeli",                # Henkel's Leaf-Tailed Gecko
    "Cyclura lewisi",                   # Grand Cayman Blue Iguana
    "Malayopython bivittatus",          # Burmese Python
    "Ara ararauna",                     # Blue and Yellow Macaw
    "Drymarchon couperi",               # Eastern Indigo Snake
    "Brachylophus fasciatus",           # Fiji Banded Iguana
    "Scopus umbretta",                  # Hamerkop
    "Chilabothrus subflavus",           # Jamaican Boa
    "Gromphadorhina portentosa",        # Madagascar Hissing Cockroach
    "Threskiornis bernieri",            # Malagasy Sacred Ibis
    "Marmaronetta angustirostris",      # Marbled Teal
    "Ara militaris",                    # Military Macaw
    "Polyplectron napoleonis",          # Palawan Peacock Pheasant
    "Atelopus zeteki",                  # Panamanian Golden Frog
    "Coendou prehensilis",              # Prehensile-Tailed Porcupine (duplicate)
    "Corucia zebrata",                  # Prehensile-Tailed Skink
    "Callosciurus prevostii",           # Prevost's Squirrel
    "Heloderma horridum exasperatum",   # Rio Fuerte Beaded Lizard
    "Ara macao",                        # Scarlet Macaw
    "Tolypeutes matacus",               # Southern Three-Banded Armadillo
    "Hydrosaurus weberi",               # Weber's Sailfin Lizard
    "Tauraco leucotis",                 # White-Cheeked Turaco
]

GBIF_SPECIES_URL  = "https://api.gbif.org/v1/species/match"
GBIF_OCCUR_URL    = "https://api.gbif.org/v1/occurrence/search"
DATA_ROOT         = os.path.join(os.path.dirname(__file__), "data")
MAX_IMAGES        = 200   # extra images to pull per species from GBIF
PAGE_SIZE         = 100   # GBIF max per request is 300, 100 is safe
RETRY_DELAY       = 2     # seconds between retries on failure


def get_taxon_key(species_name: str) -> int | None:
    """Resolve a species name to a GBIF taxonKey via the species/match endpoint."""
    try:
        resp = requests.get(GBIF_SPECIES_URL, params={"name": species_name}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        key = data.get("usageKey") or data.get("speciesKey")
        if key:
            print(f"  [{species_name}] taxonKey = {key}  (confidence {data.get('confidence')}%)")
        else:
            print(f"  [{species_name}] WARNING: no taxonKey found – matchType={data.get('matchType')}")
        return key
    except Exception as e:
        print(f"  [{species_name}] ERROR resolving taxonKey: {e}")
        return None


def fetch_image_urls(taxon_key: int, max_images: int) -> list[str]:
    """
    Page through GBIF occurrence/search for records that have StillImage media,
    collecting the first image URL from each occurrence.
    """
    urls = []
    offset = 0

    while len(urls) < max_images:
        need = min(PAGE_SIZE, max_images - len(urls))
        params = {
            "taxonKey":   taxon_key,
            "mediaType":  "StillImage",
            "limit":      need,
            "offset":     offset,
        }
        try:
            resp = requests.get(GBIF_OCCUR_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    WARNING: occurrence fetch failed (offset={offset}): {e}")
            break

        results = data.get("results", [])
        if not results:
            break  # no more records

        for occ in results:
            for media in occ.get("media", []):
                if media.get("type") == "StillImage":
                    img_url = media.get("identifier")
                    if img_url:
                        urls.append(img_url)
                        break  # one image per occurrence is enough

        # GBIF signals end-of-results
        if data.get("endOfRecords", True):
            break

        offset += len(results)

    return urls[:max_images]


def download_images(species_name: str, urls: list[str], existing_count: int) -> int:
    """Download images into ./data/<Species_name>/, skipping already-present files."""
    folder = os.path.join(DATA_ROOT, species_name.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)

    downloaded = 0
    for i, url in enumerate(tqdm(urls, desc=f"  Downloading {species_name}", leave=False)):
        # Use an index that won't collide with iNaturalist files (offset by existing_count)
        filename = os.path.join(folder, f"gbif_{i}.jpg")
        if os.path.exists(filename):
            continue  # already downloaded in a previous run
        try:
            img_resp = requests.get(url, timeout=15)
            img_resp.raise_for_status()
            content_type = img_resp.headers.get("Content-Type", "")
            if "image" not in content_type:
                continue  # skip non-image responses
            with open(filename, "wb") as f:
                f.write(img_resp.content)
            downloaded += 1
        except Exception:
            pass  # silently skip broken URLs
        time.sleep(0.05)  # be polite to GBIF servers

    return downloaded


def count_existing(species_name: str) -> int:
    folder = os.path.join(DATA_ROOT, species_name.replace(" ", "_"))
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"GBIF Dataset Builder – targeting {len(SPECIES)} species\n")

    total_added = 0
    for species in SPECIES:
        print(f"\n{'-'*60}")
        print(f"Species: {species}")

        existing = count_existing(species)
        print(f"  Existing images: {existing}")

        taxon_key = get_taxon_key(species)
        if taxon_key is None:
            print("  Skipping – could not resolve taxonKey.")
            continue

        urls = fetch_image_urls(taxon_key, MAX_IMAGES)
        print(f"  Found {len(urls)} image URLs from GBIF")

        if not urls:
            print("  No images available, skipping.")
            continue

        added = download_images(species, urls, existing)
        total_added += added
        print(f"  Added {added} new images  (folder now has ~{existing + added})")

    print(f"\n{'='*60}")  # = is ASCII-safe
    print(f"Done. Total new images added: {total_added}")