import numpy as np
import tinyarray as ta

from poisson import LinearProblem


def linear_problem_instance(discrete_system):
    """Create a linear problem from a poisson system

    Params
    ------
    discrete_system : DiscretePoisson instance
        An instance of the discrete system.

    Returns
    -------
    linear_problem : LinearProblem instance
        An instance of the linear problem with factorized LHS.
    """
    # Set all voltages to some value as this is required for the linear problem
    voltage_values = [
        (region_function, 0.0)
        for region_function in discrete_system.regions_functions["voltage"].values()
    ]

    linear_problem_instance = LinearProblem(
        discrete_poisson=discrete_system,
        voltage_val=voltage_values,
        charge_val=[],
        mixed_val=[],
        pos_voltage_mixed=None,
        pos_charge_mixed=None,
        is_charge_density=True,
        solve_problem=False,
        solver="mumps",
    )

    # Force LDU decomposition to be made
    linear_problem_instance.solve(factorize=True)

    return linear_problem_instance


def _linear_solution(
    discrete_system, linear_problem, voltages, charges, is_charge_density = True
):
    """Solution of the linear problem for the given charges and voltages

    Parameters:
    ----------
    discrete_system : DiscretePoisson instance
        An instance of the discrete system.
    linear_problem : LinearProblem instance
        An instance of the linear problem which is already factorized
    voltages : str
        Each key is the name of the gate and the value is its voltage.
    charges : dict
        Keys can be anything you wish to have.
        Values must the list of tuples. It can be of the following two cases:
        1. Providing the charge associated with each site index [(site_index, charge), ...].
        2. Providing the charge density of a charge or mixed region [(region_type, region_name, charge_density), ...]
           region type can be 'charge' or 'mixed'

    Returns
    -------
    points_voltage : np.array[:]
        The voltage value of all the sites in the discrete system.
    """
    points = discrete_system.grid.points
    indices = np.arange(points.shape[0])

    voltage_values = []
    for region_name, region_function in discrete_system.regions_functions[
        "voltage"
    ].items():
        volt = voltages[region_name]
        voltage_values.append((region_function, volt))

    charge_values = []
    mixed_values = []

    if len(charges) >= 1:
        for name, value in charges.items():

            if isinstance(value[0], tuple):
                if isinstance(value[0][0], np.int64):

                    ## code optimization - pass each unique value as a single function

                    indices = np.array(value).astype("int")[:, 0]
                    value = np.array(value)[:, 1]

                    unique_charge_indices = np.unique(indices)
                    unique_charge_values = np.unique(value)

                    for unique_charge_value in unique_charge_values:
                        charge_values.append(
                            (
                                unique_charge_indices[value == unique_charge_value],
                                unique_charge_value,
                            )
                        )

                else:
                    for element in value[0]:
                        if len(element) == 3:
                            region_type, region_name, charge_density = element

                            if region_type == "charge":

                                charge_values.append(
                                    (
                                        discrete_system.regions_functions[region_type][
                                            region_name
                                        ],
                                        charge_density,
                                    )
                                )
                            if region_type == "mixed":
                                mixed_values.append(
                                    (
                                        discrete_system.regions_functions[region_type][
                                            region_name
                                        ],
                                        charge_density,
                                    )
                                )
                        else:
                            raise ValueError(
                                "Provide information about the 1. type of region, 2. region name and 3. the charge density associated with that region."
                            )

            else:
                raise TypeError("Charge input must be a list of tuples")

    linear_problem.update(
        voltage_val=voltage_values,
        charge_val=charge_values,
        mixed_val=mixed_values,
        is_charge_density=is_charge_density,
    )

    linear_problem.solve(factorize=False)

    return linear_problem.points_voltage


def gate_potential(
    discrete_system, linear_problem, site_coords, site_indices, voltages, charges, offset = 0, grid_spacing=1
):
    
    
    """Combined potential in the two dimensional system of all voltage and charge regions

    Multiply the integrated green's function of each voltage region
    with the corresponding voltage and add them all together.

    Parameters
    ------

    discrete_system : DiscretePoisson instance
        An instance of the discrete system.
    linear_problem : LinearProblem instance
        An instance of the linear problem which is already factorized
    site_coords : np.array [n, 2]
        Coordinates for the finite discrete system based on which kwant builder is made.
    site_indices : np.array [n, ]
        Indices of the coordinates in the poisson mesh (which is stored as (n, 3) array).
        These indices can be used to return the potential in the corresponding sites.
    voltages : str
        Each key is the name of the gate and the value is its voltage.
    charges : dict
        Keys can be anything you wish to have.
        Values must the list of tuples. It can be of the following two cases:
        1. Providing the charge associated with each site index [(site_index, charge), ...].
        2. Providing the charge density of a charge or mixed region [(region_type, region_name, charge_density), ...]
           region type can be 'charge' or 'mixed'

    Returns
    -------
    Electrostatic potential : dict
        Keys contain each site coordinate and their value is the corresponding electrostatic potential.
    """
    
    solution = _linear_solution(discrete_system, linear_problem, voltages, charges)

    potential = solution[site_indices]

    potential_dict = {}

    site_tuples = map(tuple, site_coords)

    for i, site_tuple in enumerate(site_tuples):
        if site_tuple[0] < 0: 
            key = ta.array(np.round([site_tuple[0] + offset[0], 
                                     site_tuple[1] - offset[1]], 
                                    5
                                   )
                          ) 
        elif site_tuple[0] > 0: 
            key = ta.array(np.round([site_tuple[0] - offset[0], 
                                     site_tuple[1] - offset[1]], 
                                    5
                                   )
                          ) 
        
        potential_dict[key] = potential[i] ## round-off as a workaround for numerical instability # Otherwise may result in a key error when accessing this dict.

    return potential_dict
