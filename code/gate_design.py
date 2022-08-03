from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union
import json
import os
import sys
import numpy as np

from gates import rectangular_gate, half_disk_gate

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


def gate_coords(gate_config):

    """
    Returns gate vertices and gate names
    """
    
    L = gate_config["L"]
    width = gate_config["width"]
    gap = gate_config["gap"]

    R = np.round(L / np.sqrt(2))

    Y = unary_union(
        (
            Polygon(half_disk_gate(R=R, npts=3)).difference(
                Polygon(half_disk_gate(R=R - np.round(width * np.sqrt(2)), npts=3))
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

    left_1 = _gate_coords(gates[0], common=aux_rectangle_2)
    right_2 = left_1.copy()
    right_2[:, 0] *= -1
    left_2 = _gate_coords(gates[2], difference=aux_rectangle_1)
    right_1 = left_2.copy()
    right_1[:, 0] *= -1
    top_1 = _gate_coords(gates[0], difference=aux_rectangle_2, gap=gap)
    top_2 = top_1.copy()
    top_2[:, 0] *= -1
    
    gates_vertex = [
        left_1, left_2, right_1, right_2, top_1, top_2 
    ]
    
    
    # gates_vertex = [
    #     _gate_coords(gates[0], common=aux_rectangle_2),
    #     _gate_coords(gates[2], difference=aux_rectangle_1),
    #     _gate_coords(gates[2], difference=aux_rectangle_2),
    #     _gate_coords(gates[1], common=aux_rectangle_1),
    #     _gate_coords(gates[0], difference=aux_rectangle_2, gap=gap),
    #     _gate_coords(gates[1], difference=aux_rectangle_1, gap=gap),
    # ]

    gate_names = ["left_1", "left_2", "right_1", "right_2", "top_1", "top_2"]

    return gates_vertex, gate_names
