from obspy import read
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIRS = [f"./Dec{d}Data/" for d in range(25, 32)]

def check_station(args):
    station, dirs = args
    for d in dirs:
        pattern = f"{d}{station}/*.mseed"
        try:
            st = read(pattern, headonly=True)
            if len(st) != 3:
                return station, None
            # Extract human-readable station name from trace stats
            station_name = st[0].stats.station
        except Exception:
            return station, None
    return None, station_name

if __name__ == "__main__":
    stationsS = set.intersection(*(set(os.listdir(d)) for d in DATA_DIRS))
    args = [(station, DATA_DIRS) for station in stationsS]

    bad_stations = set()
    good_stations = {}  # Maps station ID -> station name

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(check_station, a): a[0] for a in args}
        for future in as_completed(futures):
            bad, name = future.result()
            if bad:
                bad_stations.add(bad)
            else:
                station_id = futures[future]
                good_stations[station_id] = name

    print(set(good_stations.values()))