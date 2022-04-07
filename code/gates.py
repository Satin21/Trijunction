import numpy as np
from collections import OrderedDict
from itertools import product
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union


def rectangular_gate(center, length, width):
    """
    Returns vertices of a gate
    """
    x, y = center

    gate = np.array(
        [
            [-width / 2, -length / 2],
            [-width / 2, length / 2],
            [width / 2, length / 2],
            [width / 2, -length / 2],
        ]
    )
    gate[:, 0] += x
    gate[:, 1] += y

    return gate


def half_disk_gate(R, center=(0, 0), npts=100, shift=0):
    """
    Return vertices of a half disk shaped polygon
    """
    x, y = center

    angles = np.linspace(0, np.pi, npts)

    xs = x + np.cos(angles) * R
    ys = y + np.sin(angles) * R

    ys[0] -= shift
    ys[-1] -= shift

    return np.vstack([xs, ys]).T


def ring_gate(R, r, center=(0, 0), npts=100):
    """
    Return vertices of a ring-shaped polygon
    Parameters:
    -----------
        R: outer radius
        r: inner radius
    """
    x, y = center

    angles = np.linspace(0, np.pi, npts)

    Xs = x + np.cos(angles) * R
    Ys = y + np.sin(angles) * R

    xs = x + np.cos(angles) * r
    ys = y + np.sin(angles) * r

    out_ring = np.vstack([Xs, Ys]).T
    in_ring = np.vstack([xs, ys]).T[::-1]

    return np.vstack([out_ring, in_ring])


def rectangular_cavity(length, width, wire_width, tunel_length, gap, extra_width):
    """
    Returns a dictionary with all the gate shapes
    """


def triangular_gates_1(area, angle, wire_width, tunel_length, gap, extra_width):
    """
    Returns the gate configuration of the trijunction
    """

    # system paramters
    triangle_length = np.sqrt(area * np.tan(angle))
    triangle_width = 2 * np.abs((triangle_length / np.tan(angle)))

    tunel_length = tunel_length
    tunel_width = wire_width

    total_length = triangle_length + tunel_length + gap
    total_width = triangle_width + 2 * gap + 2 * extra_width

    # tunel gates
    tunel_labels = ["central_tunel", "left_tunel", "right_tunel"]

    tunel_centers = np.array(
        [
            [0, tunel_length / 2],
            [-triangle_width / 4, tunel_length / 2],
            [triangle_width / 4, tunel_length / 2],
        ]
    )

    tunel_gates = dict()

    for i in range(3):
        tunel_gates[tunel_labels[i]] = rectangular_gate(
            center=tunel_centers[i], length=tunel_length, width=tunel_width
        )

    # screen gates
    screen_labels = [
        "right_screen_center",
        "right_screen_side",
        "right_screen",
        "left_screen_center",
        "left_screen_side",
        "left_screen",
    ]

    middle_width = triangle_width / 4 - tunel_width - 2 * gap
    sides_width = triangle_width / 2 - (triangle_width / 4 + tunel_width / 2 + gap)

    screen_centers = np.array(
        [
            [
                triangle_width / 8,
                tunel_length / 2,
            ],  # gate separating central and right barrier,
            [
                triangle_width / 4 + sides_width / 2 + gap + tunel_width / 2,
                tunel_length / 2,
            ],  # gate separating right barrier with rightmost gate,
            [
                triangle_width / 2 + extra_width / 2 + gap,
                total_length / 2,
            ],  # rightmost gate,
        ]
    )

    screen_sizes = np.array(
        [
            [middle_width, tunel_length],
            [sides_width, tunel_length],
            [extra_width, total_length],
        ]
    )

    screen_gate_right = np.array(
        [
            [triangle_width / 2, tunel_length + gap + 2 * gap * np.sin(angle)],
            [triangle_width / 2, total_length],
            [2 * gap * np.cos(angle), total_length],
        ]
    )

    screen_gates = dict()

    screen_gates["right_screen_triangle"] = screen_gate_right

    screen_gate_left = np.copy(screen_gate_right)
    screen_gate_left[:, 0] *= -1
    screen_gates["left_screen_triangle"] = screen_gate_left
    k = 0
    for j in range(2):
        for i in range(3):
            center = screen_centers[i]
            center[0] = center[0] * (-1) ** j
            width = screen_sizes[i][0] * (-1) ** j
            length = screen_sizes[i][1]
            screen_gates[screen_labels[k]] = rectangular_gate(
                center=center, length=length, width=width
            )
            k += 1

    extra_center = [0, total_length + extra_width / 2 + gap]
    screen_gates["top_screen"] = rectangular_gate(
        center=extra_center, length=extra_width, width=total_width
    )

    # plunger gate
    plunger_gate = np.array(
        [[-triangle_width / 2, 0], [triangle_width / 2, 0], [0, triangle_length]]
    )
    plunger_gate[:, 1] += tunel_length + gap

    plunger_gates = dict()
    plunger_gates["plunger_gate"] = plunger_gate

    # gates dictionary
    gates = {
        "plunger_gates": plunger_gates,
        "screen_gates": screen_gates,
        "tunel_gates": tunel_gates,
    }

    return gates


def triangular_gates_2(area, angle, wire_width, tunel_length, gap, extra_width):
    """
    Returns the gate configuration of the trijunction
    """

    # gate paramters
    triangle_length = np.sqrt(area * np.tan(angle))
    triangle_width = 2 * np.abs((triangle_length / np.tan(angle)))
    top_shift = np.tan(angle) * (wire_width / 2)
    tunel_length = tunel_length
    tunel_width = wire_width

    total_length = triangle_length + 2 * tunel_length + 2 * gap - top_shift
    total_width = triangle_width + 2 * gap + 2 * extra_width

    # tunel gates
    tunel_centers = np.array(
        [
            [0, triangle_length - top_shift + gap + tunel_length / 2],
            [-triangle_width / 4, tunel_length / 2],
            [triangle_width / 4, tunel_length / 2],
        ]
    )
    tunel_centers[0, 1] += tunel_length + gap

    tunel_gates = dict()
    tunel_gates["top_tunel"] = rectangular_gate(
        center=tunel_centers[0], length=tunel_length, width=tunel_width
    )
    tunel_gates["left_tunel"] = rectangular_gate(
        center=tunel_centers[1], length=tunel_length, width=tunel_width
    )
    tunel_gates["right_tunel"] = rectangular_gate(
        center=tunel_centers[2], length=tunel_length, width=tunel_width
    )

    # screen gates
    screen_length = tunel_length
    middle_width = triangle_width / 2 - wire_width - 2 * gap
    sides_width = triangle_width / 2 - (middle_width / 2 + 2 * gap + tunel_width)

    screen_centers = np.array(
        [
            [0, tunel_length / 2],
            [
                triangle_width / 4 + tunel_width / 2 + gap + sides_width / 2,
                tunel_length / 2,
            ],
            [
                -(triangle_width / 4 + tunel_width / 2 + gap + sides_width / 2),
                tunel_length / 2,
            ],
            [triangle_width / 2 + extra_width / 2 + gap, total_length / 2],
            [-triangle_width / 2 - extra_width / 2 - gap, total_length / 2],
        ]
    )

    screen_gate_right = np.array(
        [
            [triangle_width / 2, tunel_length + gap * (3 / 2) + gap * np.tan(angle)],
            [triangle_width / 2, total_length],
            [tunel_width / 2 + gap, total_length],
            [tunel_width / 2 + gap, total_length - tunel_length - gap / 2],
        ]
    )

    screen_gate_left = np.copy(screen_gate_right)
    screen_gate_left[:, 0] *= -1

    screen_gates = {
        "left_screen_triangle": screen_gate_left,
        "right_screen_triangle": screen_gate_right,
        "central_screen_gate": rectangular_gate(
            center=screen_centers[0], length=screen_length, width=middle_width
        ),
        "left_screen_side": rectangular_gate(
            center=screen_centers[1], length=screen_length, width=sides_width
        ),
        "right_screen_side": rectangular_gate(
            center=screen_centers[2], length=screen_length, width=sides_width
        ),
        "left_screen": rectangular_gate(
            center=screen_centers[3], length=total_length, width=extra_width
        ),
        "right_screen": rectangular_gate(
            center=screen_centers[4], length=total_length, width=extra_width
        ),
    }

    # plunger gate
    plunger_gate = np.array(
        [
            [-triangle_width / 2, 0],
            [triangle_width / 2, 0],
            [wire_width / 2, triangle_length - top_shift],
            [-wire_width / 2, triangle_length - top_shift],
        ]
    )
    plunger_gate[:, 1] += tunel_length + gap
    plunger_gates = dict()
    plunger_gates["plunger_gate"] = plunger_gate

    gates = {
        "plunger_gates": plunger_gates,
        "screen_gates": screen_gates,
        "tunel_gates": tunel_gates,
    }

    return gates


def ring_gates(R, wire_width, gap, tunel_length):

    y_shift = tunel_length + gap
    y_c = tunel_length / 2
    r = R - wire_width

    ring_plunger = ring_gate(R=R, r=r, center=(0, 0))
    central_tunnel = rectangular_gate(
        center=(0, R + gap + y_c), length=tunel_length, width=wire_width
    )

    center = (0, max(ring_plunger[:, 1]) - (R - r) / 2)
    central_ring_tunnel = rectangular_gate(
        center=center, length=R - r, width=wire_width
    )

    xmin = min(ring_plunger[:, 0]) - 10
    xmax = -xmin

    ymin = min(ring_plunger[:, 1])
    ymax = max(central_tunnel[:, 1])

    system_vertices = list(product((xmin, xmax), (ymin, ymax)))
    system_vertices = np.array(system_vertices)
    total_system = system_vertices[
        np.lexsort((system_vertices[:, 0], system_vertices[:, 1]))
    ][[0, 1, 3, 2]]

    ring = Polygon(ring_plunger).difference(Polygon(central_ring_tunnel).buffer(gap))
    central_tunnel = Polygon(central_tunnel)
    central_ring_tunnel = Polygon(central_ring_tunnel)

    accumulation_gates_union = unary_union(
        (ring.buffer(gap), central_tunnel.buffer(gap), central_ring_tunnel.buffer(gap))
    )

    screening_gate = Polygon(total_system).difference(accumulation_gates_union)

    screening_coords = []
    for poly in list(screening_gate.geoms):
        screening_coords.append(np.array(list(poly.exterior.coords)))

    left_ring_coords = np.array(list(list(ring.geoms)[0].exterior.coords))
    right_ring_coords = np.array(list(list(ring.geoms)[1].exterior.coords))
    central_tunnel_coords = np.array(list(central_tunnel.exterior.coords))
    central_ring_tunnel_coords = np.array(list(central_ring_tunnel.exterior.coords))

    gates_name = ["left_ring", "right_ring", "central_tunnel", "central_ring_tunnel"]
    gates_vertex = [
        left_ring_coords,
        right_ring_coords,
        central_tunnel_coords,
        central_ring_tunnel_coords,
    ]

    for i, coord in enumerate(screening_coords):
        gates_name.append("screening" + str(i))
        gates_vertex.append(coord)

    return gates_name, gates_vertex
