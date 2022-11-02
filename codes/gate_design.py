import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import split
from itertools import product
from collections import OrderedDict


def gate_coords(config):

    """Find the gate coordinates that defines the trijunction

    L: Channel length
    gap: Gap between gates
    channel_width: width of the channel
    angle: Angle the channel makes with the x-axis [in radians]

    Returns:
    Gates vertices and their names
    boundary: coordinates of the four corners of the system envelope
    channel centers: Coordinates of the 2DEG at which nanowires can be connected

    """

    L = config["gate"]["L"]
    gap = config["gate"]["gap"]
    channel_width = config["gate"]["channel_width"]
    angle = config["gate"]["angle"]

    A = lambda x: np.array([[-np.cos(x), np.sin(x)], [np.sin(x), np.cos(x)]])

    point = np.array([0, 1])

    rotated_c = (point @ A(angle)) @ (np.eye(2) * L)

    tail = np.round(rotated_c.reshape((2, 1)) * np.array([[-1, 0]]).T, 2)
    head = np.round(rotated_c.reshape((2, 1)) * np.array([[0, 1]]).T, 2)

    left, right, top = (
        LineString([Point(*tail), Point(*head)]),
        LineString([Point(*(tail * -1)), Point(*head)]),
        LineString([Point(*head), Point([0, L + head[1]])]),
    )

    def shift(x, adjustment):
        if not isinstance(x, np.ndarray): x = np.array(x.coords)
        x[:, 0] += adjustment
        return x

    distance_to_axis = channel_width / 2

    lcoords = shift(left, -distance_to_axis)
    lcoords_p = shift(left, distance_to_axis)
    rcoords = shift(right, distance_to_axis)
    rcoords_p = shift(right, -distance_to_axis)
    tcoords = shift(top, -distance_to_axis)
    tcoords_p = shift(top, distance_to_axis)

    intersection = list(
        LineString(lcoords_p).intersection(LineString(rcoords_p)).coords
    )

    lcoords_p[1] = intersection[0]
    rcoords_p[1] = intersection[0]

    coords = np.vstack(
        (
            lcoords,
            tcoords,
            tcoords_p[::-1],
            rcoords[::-1],
            rcoords_p,
            lcoords_p[::-1],
            lcoords[0],
        )
    )
    trijunction = Polygon(coords)

    xmin, xmax, ymin, ymax = (
        coords[:, 0].min(),
        coords[:, 0].max(),
        coords[:, 1].min(),
        coords[:, 1].max(),
    )

    rectangle = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])

    gates = Polygon(rectangle).difference(trijunction)

    gates = np.array(gates.geoms)
    area = [gate.area for gate in gates]
    gates = gates[np.argsort(area)]

    coords = np.array(gates[1].exterior.coords)
    y = np.unique(coords[:, 1])
    assert len(y) == 3
    bottom_top_threshold = y[1] + gap / 2

    splitter = LineString(
        [
            Point(point)
            for point in product(np.unique(coords[:, 0]), [bottom_top_threshold])
        ]
    )

    top, bottom = list(gates[1].difference(splitter.buffer(gap / 2)).geoms)
    
    grid_spacing = config['device']['grid_spacing']['gate']
    
    left_1 = np.round(np.array(bottom.exterior.coords))
    top_1 = np.round(np.array(top.exterior.coords))
    top_2 = top_1.copy() @ [[-1, 0], [0, 1]]
    right_2 = left_1.copy() @ [[-1, 0], [0, 1]]

    coords = np.array(gates[0].exterior.coords)

    splitter = LineString(
        [Point(point) for point in product([0.0], np.unique(coords[:, 1]))]
    )

    left_2, _ = list(gates[0].difference(splitter.buffer(gap / 2)).geoms)

    left_2 = np.round(np.array(left_2.exterior.coords))
    right_1 = left_2.copy() @ [[-1, 0], [0, 1]]

    gates_vertex = [left_1, left_2, right_1, right_2, top_1, top_2]
    
    gate_names = ["left_1", "left_2", "right_1", "right_2", "top_1", "top_2"]
    assert min(left_1[:, 0]) == min(top_1[:, 0])
    assert max(right_2[:, 0]) == max(top_2[:, 0])

    boundaries = OrderedDict(
        xmin=min(left_1[:, 0]),
        xmax=max(right_2[:, 0]),
        ymin=min(left_1[:, 1]),
        ymax=max(top_1[:, 1]),
    )

    channel_center = OrderedDict(
        left=np.hstack(tail),
        right=np.hstack(tail * -1),
        top=np.hstack([0, max(top_1[:, 1])]),
    )

    return (gates_vertex, 
            gate_names,
            boundaries,
            channel_center)


def consistent_grid(coords, grid_spacing):
    """
    Avoid numerical errors due to very small decimal values in the grid coords those are inconsistent with the
    grid spacing.
    """
    if np.any(coords % grid_spacing):
        coords -= coords % grid_spacing
    return coords
