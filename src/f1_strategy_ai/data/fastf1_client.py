import os
from pathlib import Path
import fastf1


def init_fastf1_cache(cache_dir_name: str = "f1_cache") -> None:
    #init_fastf1_cache: It creates a local directory at the project root to store downloaded data and avoid repeated API requests
    project_root = Path(__file__).resolve().parents[3]
    cache_path = project_root / cache_dir_name
    os.makedirs(cache_path, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))


def load_race_session(year: int, grand_prix: str, session_type: str = "R"):
    #load_race_session: It downloads and loads an F1 session
    #Args: year, grand_prix, session_type (R/Q)

    init_fastf1_cache()
    print(f"Session: {year} {grand_prix} [{session_type}]")
    session = fastf1.get_session(year, grand_prix, session_type)
    session.load(telemetry=True, laps=True)
    return session

