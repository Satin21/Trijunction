from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union
import json
import os
import sys
import numpy as np

from gates import rectangular_gate, half_disk_gate

sys.path.append(os.path.realpath(sys.path[0] + "/.."))
from rootpath import ROOT_DIR

filepath = os.path.join(ROOT_DIR, "code/")


filename = "config.json"
with open(filepath + filename, "r") as infile:
    config = json.load(infile)

gate_config = config["gate"]


def _gate_coords(obj, difference=None, common=None, gap=None):

    if type(common) == np.ndarray:
        return np.round(
            np.array(list(obj.intersection(Polygon(common)).exterior.coords)), 2
        )

    else:
        if gap is not None:
            return np.round(
                np.array(
                    list(
                        obj.difference(Polygon(difference).buffer(gap)).exterior.coords
                    )
                ),
                2,
            )
        else:
            return np.round(
                np.array(list(obj.difference(Polygon(difference)).exterior.coords)), 2
            )


def gate_coords(
    L: float = gate_config["L"],
    width: float = gate_config["width"],
    gap: float = gate_config["gap"],
):

    """
    Returns gate vertices and gate names
    """

    R = L / np.sqrt(2)

    Y = unary_union(
        (
            Polygon(half_disk_gate(R=R, npts=3)).difference(
                Polygon(half_disk_gate(R=R - width * np.sqrt(2), npts=3))
            ),
            Polygon(
                rectangular_gate(center=(0, R + L / 2 - width), width=width, length=L)
            ),
        )
    )

    gates = Polygon(
        rectangular_gate(
            center=(0, (R + L - width) / 2), length=R + L - width - 1, width=2 * R
        )
    ).difference(Y)

    aux_rectangle_1 = rectangular_gate(
        length=R + 2 * gap, width=R + gap, center=(R / 2, R / 2 - width / 2)
    )
    aux_rectangle_2 = rectangular_gate(
        length=R + 2 * gap, width=R + gap, center=(-R / 2, R / 2 - width / 2)
    )

    gates = gates.geoms
    gates_vertex = [
        _gate_coords(gates[0], common=aux_rectangle_2),
        _gate_coords(gates[2], difference=aux_rectangle_1),
        _gate_coords(gates[2], difference=aux_rectangle_2),
        _gate_coords(gates[1], common=aux_rectangle_1),
        _gate_coords(gates[0], difference=aux_rectangle_2, gap=gap),
        _gate_coords(gates[1], difference=aux_rectangle_1, gap=gap),
    ]

    gate_names = ["left_1", "left_2", "right_1", "right_2", "top_1", "top_2"]

    return gates_vertex, gate_names
