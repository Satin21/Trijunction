import numpy as np
import sys, os

<<<<<<< HEAD:codes/discretize.py
sys.path.append("/home/tinkerer/spin-qubit/")
=======
sys.path.append(os.path.realpath("./../spin-qubit/"))

sys.path.append("/home/srangaswamykup/trijunction_design/spin-qubit/")
>>>>>>> master:code/discretize.py

from layout import (
    Layout,
    OverlappingGateLayer,
    PlanarGateLayer,
    SimpleChargeLayer,
    TwoDEGLayer,
)


def check_grid(A, B):
    if A % B:
        return A % B
    return B


def discretize_heterostructure(config, boundaries, gate_vertices, gate_names):
    """
    Parameters
    ----------
    config: dictionary
    boundaries: dictionary
    gate_vertices: ?
    gate_names: np.ndarray ?

    Returns
    -------
    Poisson system
    """

    device_config = config["device"]
    grid_spacing = device_config["grid_spacing"]
    thickness = device_config["thickness"]
    permittivity = device_config["permittivity"]

    total_width = boundaries["xmax"] - boundaries["xmin"]
    total_length = boundaries["ymax"] - boundaries["ymin"]

    # https://arxiv.org/pdf/2105.10437.pdf

    # change: add an extra layer for the substrate
    margin = (50, 50, 50)

    layout = Layout(
        total_width,
        total_length,
        grid_width_air=grid_spacing["air"],
        margin=margin,
        shift=(0, total_length / 2, 0),
    )

    # Al_{0.1}In_{0.9}Sb_4 Pm
    layout.add_layer(
        SimpleChargeLayer(
            "substrate",
            thickness["substrate"],
            permittivity["substrate"],  # TODO: yet to find exact value
            grid_spacing["substrate"],
        ),
    )

    layout.add_layer(
        TwoDEGLayer(
            "twoDEG",
            thickness["twoDEG"],
            permittivity["twoDEG"],
            grid_spacing["twoDEG"],
        ),
        center=True,
    )

    height = thickness["twoDEG"] / 2

    def check_grid(A, B):
        if A % B:
            return A % B
        return B

    lattice_constant = check_grid(thickness["dielectric"], grid_spacing["dielectric"])

    layout.add_layer(
        SimpleChargeLayer(
            "Al2O3",
            thickness["dielectric"],
            permittivity["Al2O3"],
            lattice_constant,
        )
    )

    height += thickness["dielectric"]

    lattice_constant = check_grid(thickness["gates"], grid_spacing["gate"])

    layout.add_layer(
        OverlappingGateLayer(
            thickness["gates"],
            permittivity["metal"],
            lattice_constant,
            layer_name=gate_names,
            gate_objects=gate_vertices,
            z_bottom=height,
            fix_overlap=True,
        )
    )

    height += thickness["gates"]

    lattice_constant = check_grid(thickness["dielectric"], grid_spacing["dielectric"])

    layout.add_layer(
        SimpleChargeLayer(
            "Al2O3_2",
            thickness["dielectric"],
            permittivity["Al2O3"],
            lattice_constant,
            z_bottom=height,
            fix_overlap=False,
        )
    )

    height += thickness["dielectric"]
    thickness_accumulation_gate = 2

    grid_spacing = check_grid(thickness_accumulation_gate, grid_spacing["gate"])

    layout.add_layer(
        PlanarGateLayer(
            "global_accumul",
            thickness_accumulation_gate,
            permittivity["metal"],
            grid_spacing,
            z_bottom=height,
        )
    )

    height += thickness_accumulation_gate

    ## Surround the device with gates for applying Dirichlet boundary condition

    thickness_gate = device_config["thickness"]["gates"]
    grid_spacing_gate = device_config["grid_spacing"]["gate"] * 10
    thickness_gate = thickness_gate + (grid_spacing_gate / 10)

    # height = sum(layer.thickness for layer in layout.layers)

    assert -(thickness["substrate"] + thickness["twoDEG"] / 2) == layout.z_bottom

    zmin = layout.z_bottom - margin[2]
    zmax = height + margin[2]

    xmin, xmax, ymin, ymax = (
        -total_width / 2 - margin[0],
        total_width / 2 + margin[0],
        -margin[1],
        total_length + margin[1],
    )

    centers = np.array(
        [
            [
                [
                    xmin - (thickness_gate / 2),
                    ymin + (ymax - ymin) / 2,
                    zmin + (zmax - zmin) / 2,
                ]
            ],
            [
                [
                    xmax + (thickness_gate / 2),
                    ymin + (ymax - ymin) / 2,
                    zmin + (zmax - zmin) / 2,
                ]
            ],
            [[0.0, ymin - (thickness_gate / 2), zmin + (zmax - zmin) / 2]],
            [[0.0, ymax + (thickness_gate / 2), zmin + (zmax - zmin) / 2]],
            [[0.0, ymin + (ymax - ymin) / 2, zmin - (thickness_gate / 2)]],
            [[0.0, ymin + (ymax - ymin) / 2, zmax + (thickness_gate / 2)]],
        ]
    )

    lengths = [
        thickness_gate,
        thickness_gate,
        xmax - xmin,
        xmax - xmin,
        xmax - xmin,
        xmax - xmin,
    ]
    widths = [
        ymax - ymin,
        ymax - ymin,
        thickness_gate,
        thickness_gate,
        ymax - ymin,
        ymax - ymin,
    ]
    heights = [
        zmax - zmin,
        zmax - zmin,
        zmax - zmin,
        zmax - zmin,
        thickness_gate,
        thickness_gate,
    ]

    # return centers, lengths, widths, heights
    layout.height = sum(layer.thickness for layer in layout.layers)

    names = []
    for i, data in enumerate(zip(centers, lengths, widths, heights)):
        center, length, width, height = data

        name = "dirichlet_" + str(i)

        layout.add_layer(
            PlanarGateLayer(
                name,
                thickness_gate,
                device_config["permittivity"]["metal"],
                grid_spacing_gate,
                fix_overlap=True,
                center=center,
                length=length,
                width=width,
                height=height,
                z_bottom=None,
            )
        )

    return layout.build()
