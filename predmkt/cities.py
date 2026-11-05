"""Kalshi daily-high series, the station each settles on (NWS climate report site), and local standard time.

Settlement day = local STANDARD time midnight-to-midnight (even in summer), which is why Kalshi closes at
05:00Z for New York in October. lat/lon are the airport/park station, used for the forecast grid cell.
"""
CITIES = {
    "NYC": dict(series="KXHIGHNY", iem="KNYC", lat=40.7789, lon=-73.9692, lst=-5),
    "CHI": dict(series="KXHIGHCHI", iem="KMDW", lat=41.7868, lon=-87.7522, lst=-6),
    "MIA": dict(series="KXHIGHMIA", iem="KMIA", lat=25.7932, lon=-80.2906, lst=-5),
    "AUS": dict(series="KXHIGHAUS", iem="KAUS", lat=30.1945, lon=-97.6699, lst=-6),
    "LAX": dict(series="KXHIGHLAX", iem="KLAX", lat=33.9425, lon=-118.4081, lst=-8),
    "DEN": dict(series="KXHIGHDEN", iem="KDEN", lat=39.8561, lon=-104.6737, lst=-7),
    "PHL": dict(series="KXHIGHPHIL", iem="KPHL", lat=39.8744, lon=-75.2424, lst=-5),
}
MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless"]
