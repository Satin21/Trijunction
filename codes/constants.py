scale = 10e-9
majorana_pair_indices = {
    "left-right": [0, 1],
    "left-top": [0, 2],
    "right-top": [1, 2],
}
voltage_keys = {
    "left_1": 0,
    "left_2": 0,
    "right_1": 1,
    "right_2": 1,
    "top_1": 2,
    "top_2": 2,
    "global_accumul": 3,
}


def phase_pairs(pair, phi):
    if pair == "right-top":
        return {"phi2": phi, "phi1": 0}
    if pair == "left-top":
        return {"phi2": phi, "phi1": 0}
    if pair == "left-right":
        return {"phi1": phi, "phi2": 0}
