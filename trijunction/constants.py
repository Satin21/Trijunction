scale = 10e-9

pairs = ["right-top", "left-top", "left-right"]
sides = ["left", "right", "top"]

rounding_limit = 3
majorana_pair_indices = dict(zip(pairs, [[1, 2], [0, 2], [0, 1]]))

topological_gap = 0.0003249887255233884

voltage_keys = {
    "left_1": 0,
    "left_2": 0,
    "right_1": 1,
    "right_2": 1,
    "top_1": 2,
    "top_2": 2,
    "global_accumul": 3,
}

# Bottom of each transverse band
bands = [
    0.0023960204649275973,
    0.009605416498312178,
    0.020395040147213304,
    0.03312226901926766,
    0.045849497891322026,
    0.056639121540223145,
    0.06384851757360771,
]

default = {"rhobeg": 1e-3}
