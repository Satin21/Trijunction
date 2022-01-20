import numpy as np
from collections import OrderedDict


def rectangular_gate(center, length, width):
    """
    Returns vertices of a gate
    """
    x, y = center

    gate = np.array(
        [
            [-width/2, -length/2],
            [-width/2, length/2],
            [width/2, length/2],
            [width/2, -length/2]
        ]
    )
    gate[:, 0] += x
    gate[:, 1] += y

    return gate


def triangular_gates_1(area, angle, wire_width, tunel_length, gap, extra_width):
    """
    Returns the gate configuration of the trijunction
    """

    # system paramters
    triangle_length = np.sqrt(area*np.tan(angle))
    triangle_width = 2 * np.abs((triangle_length/np.tan(angle)))

    tunel_length = tunel_length
    tunel_width = wire_width

    total_length = triangle_length + tunel_length + gap
    total_width = triangle_width + 2* gap + 2*extra_width
    
    # tunel gates
    tunel_labels = [
        'central_tunel',
        'left_tunel',
        'right_tunel'
    ]

    tunel_centers = np.array(
        [
            [0, tunel_length/2],
            [- triangle_width/4, tunel_length/2],
            [triangle_width/4, tunel_length/2]
        ]
    )
    
    tunel_gates = dict()
    
    for i in range(3):
        tunel_gates[tunel_labels[i]] = rectangular_gate(center=tunel_centers[i], length=tunel_length, width=tunel_width)

    # screen gates
    screen_labels = [
        'right_screen_center',
        'right_screen_side',
        'right_screen',
        'left_screen_center',
        'left_screen_side',
        'left_screen'
    ]

    middle_width = triangle_width/4 - tunel_width - 2 * gap
    sides_width = triangle_width/2 - (triangle_width/4 + tunel_width/2 + gap)
    
    screen_centers = np.array(
        [
            [triangle_width/8, tunel_length/2], # gate separating central and right barrier,
            [triangle_width/4 + sides_width/2 + gap + tunel_width/2, tunel_length/2], # gate separating right barrier with rightmost gate,
            [triangle_width/2 + extra_width/2 + gap, total_length/2], # rightmost gate,
        ]
    )

    screen_sizes = np.array(
        [
            [middle_width, tunel_length],
            [sides_width, tunel_length],
            [extra_width, total_length]
        ]
    )
    
    screen_gate_right = np.array(
        [
            [triangle_width/2, tunel_length + gap + 2*gap*np.sin(angle)],
            [triangle_width/2, total_length],
            [2*gap*np.cos(angle), total_length],
        ]
    )
    
    screen_gates = dict()
    
    screen_gates['right_screen_triangle'] = screen_gate_right

    screen_gate_left = np.copy(screen_gate_right)
    screen_gate_left[:, 0] *= -1
    screen_gates['left_screen_triangle'] = screen_gate_left
    k = 0
    for j in range(2):
        for i in range(3):
            center = screen_centers[i]
            center[0] = center[0] * (-1)**j
            width = screen_sizes[i][0] * (-1)**j
            length = screen_sizes[i][1]
            screen_gates[screen_labels[k]] = rectangular_gate(center=center, length=length, width=width)
            k += 1
    
    extra_center = [0, total_length + extra_width/2 + gap]
    screen_gates['top_screen'] = rectangular_gate(center=extra_center, length=extra_width, width=total_width)
            
    # plunger gate
    plunger_gate = np.array(
        [
            [-triangle_width/2, 0],
            [triangle_width/2, 0],
            [0, triangle_length]
        ]
    )
    plunger_gate[:, 1] += tunel_length + gap

    plunger_gates = dict()
    plunger_gates['plunger_gate'] = plunger_gate

    # gates dictionary
    gates = {'plunger_gates': plunger_gates,
             'screen_gates': screen_gates,
             'tunel_gates': tunel_gates}

    return gates


def triangular_gates_2(area, angle, wire_width, tunel_length, gap, extra_width):
    """
    Returns the gate configuration of the trijunction
    """

    # gate paramters
    triangle_length = np.sqrt(area*np.tan(angle))
    triangle_width = 2 * np.abs((triangle_length/np.tan(angle)))
    top_shift = np.tan(angle)*(wire_width/2)
    tunel_length = 2
    tunel_width = wire_width

    total_length = triangle_length + 2 * tunel_length + 2 * gap - top_shift
    total_width = triangle_width + 2* gap + 2*extra_width

    # tunel gates
    tunel_centers = np.array(
        [
            [0, triangle_length - top_shift + gap + tunel_length/2],
            [- triangle_width/4, tunel_length/2],
            [triangle_width/4, tunel_length/2]
        ]
    )
    tunel_centers[0, 1] += tunel_length + gap

    tunel_gates = dict()
    tunel_gates['top_tunel'] = rectangular_gate(center=tunel_centers[0],
                                      length=tunel_length, 
                                      width=tunel_width)
    tunel_gates['left_tunel'] = rectangular_gate(center=tunel_centers[1],
                                       length=tunel_length,
                                       width=tunel_width)
    tunel_gates['right_tunel'] = rectangular_gate(center=tunel_centers[2],
                                        length=tunel_length,
                                        width=tunel_width)

    # screen gates
    screen_length = tunel_length
    middle_width = triangle_width/2 - wire_width - 2 * gap
    sides_width = triangle_width/2 - (middle_width/2 + 2*gap + tunel_width)

    screen_centers = np.array(
        [
            [0, tunel_length/2],
            [triangle_width/4 + tunel_width/2 + gap + sides_width/2, tunel_length/2],
            [-(triangle_width/4 + tunel_width/2 + gap + sides_width/2), tunel_length/2],
            [triangle_width/2 + extra_width/2 + gap, total_length/2],
            [- triangle_width/2 - extra_width/2 - gap, total_length/2]
        ]
    )

    screen_gate_right = np.array(
        [
            [triangle_width/2, tunel_length + gap * (3/2) + gap * np.tan(angle)],
            [triangle_width/2, total_length],
            [tunel_width/2 + gap, total_length],
            [tunel_width/2 + gap, total_length - tunel_length - gap/2]
        ]
    )

    screen_gate_left = np.copy(screen_gate_right)
    screen_gate_left[:, 0] *= -1
    
    screen_gates = {'left_screen_triangle': screen_gate_left,
                    'right_screen_triangle': screen_gate_right,
                    'central_screen_gate': rectangular_gate(center=screen_centers[0],
                                                            length=screen_length,
                                                            width=middle_width),
                    'left_screen_side': rectangular_gate(center=screen_centers[1],
                                                         length=screen_length,
                                                         width=sides_width),
                    'right_screen_side': rectangular_gate(center=screen_centers[2],
                                                          length=screen_length,
                                                          width=sides_width),
                    'left_screen': rectangular_gate(center=screen_centers[3],
                                                    length=total_length,
                                                    width=extra_width),
                    'right_screen': rectangular_gate(center=screen_centers[4],
                                                     length=total_length,
                                                     width=extra_width)
    }

    # plunger gate
    plunger_gate = np.array(
        [
            [-triangle_width/2, 0],
            [triangle_width/2, 0],
            [wire_width/2, triangle_length-top_shift],
            [-wire_width/2, triangle_length-top_shift]
        ]
    )
    plunger_gate[:, 1] += tunel_length + gap
    plunger_gates = dict()
    plunger_gates['plunger_gate'] = plunger_gate

    gates = {'plunger_gates': plunger_gates,
             'screen_gates': screen_gates,
             'tunel_gates': tunel_gates}

    return gates

def triangular_gates_3(area, angle, wire_width, tunel_length, gap, extra_width):
    """
    Returns the gate configuration of the trijunction
    """

    # gate paramters
    triangle_length = np.sqrt(area*np.tan(angle))
    triangle_width = 2 * np.abs((triangle_length/np.tan(angle)))
    top_shift = np.tan(angle)*(wire_width/2)
    tunel_length = 2
    tunel_width = wire_width

    total_length = triangle_length + 2 * tunel_length + 2 * gap - top_shift
    total_width = triangle_width + 2* gap + 2*extra_width
    center = np.abs((triangle_length/np.tan(angle))*0.5)
    edges = [center + wire_width/2, center - wire_width/2]
    
    def diagonal_line(x, angle=angle):
        L = total_length - tunel_length - gap/2
        return L - np.tan(angle) * (x - (tunel_width/2 + gap))
    
    
    screen_gate_1 = np.array(
        [
            [triangle_width/2, tunel_length + gap * (3/2) + gap * np.tan(angle)],
            [triangle_width/2, total_length],
            [edges[0] + gap, total_length],
            [edges[0] + gap, diagonal_line(edges[0] + gap)],
            [triangle_width/2, tunel_length + gap * (3/2) + gap * np.tan(angle)]
        ]
    )


    tunel_gate_1 = np.array(
            [
                [edges[0], diagonal_line(edges[0])],
                [edges[0], total_length],
                [edges[1], total_length],
                [edges[1], diagonal_line(edges[1])],
                [edges[0], diagonal_line(edges[0])],
            ]
        )

    screen_gate_2 = np.array(
            [
                [edges[1] - gap, diagonal_line(edges[1] - gap)],
                [edges[1] - gap, total_length],
                [tunel_width/2 + gap, total_length],
                [tunel_width/2 + gap, total_length - tunel_length - gap/2],            
                [edges[1] - gap, diagonal_line(edges[1] - gap)]
            ]
        )
    

    # tunel gates
    tunel_centers = np.array(
        [
            [0, triangle_length - top_shift + gap + tunel_length/2],
            [- triangle_width/4, tunel_length/2],
            [triangle_width/4, tunel_length/2]
        ]
    )
    tunel_centers[0, 1] += tunel_length + gap

    tunel_gates = dict()
    tunel_gates['top_tunel'] = rectangular_gate(center=[0, triangle_length - top_shift + 2*gap + 3/2*tunel_length],
                                                length=tunel_length, 
                                                width=tunel_width)
    tunel_gates['right_tunel'] = tunel_gate_1
    tunel_gate_2 = np.copy(tunel_gate_1)
    tunel_gate_2[:, 0] *= -1
    tunel_gates['left_tunel'] = tunel_gate_2

    # screen gates
    screen_centers = np.array(
        [
            [0, tunel_length/2],
            [triangle_width/2 + extra_width/2 + gap, total_length/2],
            [- triangle_width/2 - extra_width/2 - gap, total_length/2]
        ]
    )

    screen_gates = {}
    screen_gates['right_side_screen'] = screen_gate_1
    screen_gates['right_middle_screen'] = screen_gate_2
    
    screen_gate_3 = np.copy(screen_gate_1)
    screen_gate_3[:, 0] *= -1
    
    screen_gate_4 = np.copy(screen_gate_2)
    screen_gate_4[:, 0] *= -1
    
    screen_gates['left_side_screen'] = screen_gate_3
    screen_gates['left_middle_screen'] = screen_gate_4
    
    screen_gates['bottom_screen'] = rectangular_gate(center=screen_centers[0],
                                                  length=tunel_length, 
                                                  width=triangle_width)
    screen_gates['right_screen'] = rectangular_gate(center=screen_centers[1],
                                                   length=total_length, 
                                                   width=extra_width)
    screen_gates['left_screen'] = rectangular_gate(center=screen_centers[2],
                                                   length=total_length, 
                                                   width=extra_width)


    # plunger gate
    plunger_gate = np.array(
        [
            [-triangle_width/2, 0],
            [triangle_width/2, 0],
            [wire_width/2, triangle_length-top_shift],
            [-wire_width/2, triangle_length-top_shift]
        ]
    )
    plunger_gate[:, 1] += tunel_length + gap
    plunger_gates = dict()
    plunger_gates['plunger_gate'] = plunger_gate

    gates = {'plunger_gates': plunger_gates,
             'screen_gates': screen_gates,
             'tunel_gates': tunel_gates}

    return gates

