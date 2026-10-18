"""Downloads (free, public, Iowa Environmental Mesonet):
- ASOS routine hourly METAR observations (report_type=3), temperature / dew point, UTC.
- NWS GFS MOS (MAV) bulletins for the same stations: the operational statistical forecast,
  used as a professional baseline (never as a feature).
Files land in data/raw/{asos,mav}/ (not committed). Polite: sequential requests with pauses."""
import sys, time, urllib.request
from pathlib import Path
from .stations import CITY_STATION

IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request"


def get(url, out, tries=5):
    if out.exists() and out.stat().st_size > 100:
        return
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                out.write_bytes(r.read()); time.sleep(2); return
        except Exception as e:
            print("retry", out.name, e); time.sleep(10 * (i + 1))
    raise RuntimeError(url)


def main(y0=2021, y1=2026):
    for st in CITY_STATION.values():
        for y in range(y0, y1 + 1):
            get(f"{IEM}/asos.py?station={st}&data=tmpf&data=dwpf&year1={y}&month1=1&day1=1&year2={y + 1}&month2=1&day2=1"
                f"&tz=Etc/UTC&format=onlycomma&missing=empty&trace=empty&report_type=3",
                Path(f"data/raw/asos/{st}_{y}.csv"))
            get(f"{IEM}/mos.py?station=K{st}&model=GFS&sts={y}-01-01T00:00Z&ets={y + 1}-01-01T00:00Z&format=csv",
                Path(f"data/raw/mav/K{st}_{y}.csv"))
            print(st, y, flush=True)


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
