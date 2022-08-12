import numpy as np
import sys, os

ROOT_DIR = os.path.realpath(sys.path[0] + '/../')

# pre-defined functions from spin-qubit repository
sys.path.append(ROOT_DIR + "/spin-qubit/")
from layout import (
    Layout,
    OverlappingGateLayer,
    PlanarGateLayer,
    SimpleChargeLayer,
    TwoDEGLayer,
)


from gate_design import gate_coords


def discretize_heterostructure(config, boundaries):

    device_config = config["device"]
    grid_spacing = device_config["grid_spacing"]
    thickness = device_config["thickness"]
    permittivity = device_config["permittivity"]
    gate_config = config["gate"]

    total_width = 2 * boundaries["xmax"]
    total_length = boundaries["ymax"]

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
        SimpleChargeLayer(
            "twoDEG",
            thickness["twoDEG"],
            permittivity["twoDEG"],
            grid_spacing["twoDEG"],
        ),
        center=True,
    )

    height = thickness["twoDEG"] / 2

    layout.add_layer(
        SimpleChargeLayer(
            "Al2O3",
            thickness["dielectric"],
            permittivity["Al2O3"],
            grid_spacing["dielectric"],
        )
    )

    height += thickness["dielectric"]

    gate_vertices, gate_names = gate_coords(gate_config)

    layout.add_layer(
        OverlappingGateLayer(
            thickness["gates"],
            permittivity["metal"],
            grid_spacing["gate"],
            layer_name=gate_names,
            gate_objects=gate_vertices,
            z_bottom=height,
            fix_overlap=True,
        )
    )

    height += thickness["gates"]
    layout.add_layer(
        SimpleChargeLayer(
            "Al2O3_2",
            thickness["dielectric"],
            permittivity["Al2O3"],
            grid_spacing["dielectric"],
            z_bottom=height,
            fix_overlap=False,
        )
    )

    height += thickness["dielectric"]
    thickness_accumulation_gate = 2
    layout.add_layer(
        PlanarGateLayer(
            "global_accumul",
            thickness_accumulation_gate,
            permittivity["metal"],
            grid_spacing["gate"],
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
            [[xmin - (thickness_gate / 2), 0.0, zmin + (zmax - zmin) / 2]],
            [[xmax + (thickness_gate / 2), 0.0, zmin + (zmax - zmin) / 2]],
            [[0.0, ymin - (thickness_gate / 2), zmin + (zmax - zmin) / 2]],
            [[0.0, ymax + (thickness_gate / 2), zmin + (zmax - zmin) / 2]],
            [[0.0, 0.0, zmin - (thickness_gate / 2)]],
            [[0.0, 0.0, zmax + (thickness_gate / 2)]],
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
