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


def gates_trijunction(area, angle, wire_width, gap, extra_width):
    """
    Returns the gate configuration of the trijunction
    """

    # system paramters
    triangle_length = np.sqrt(area*np.tan(angle))
    triangle_width = np.abs((triangle_length/np.tan(angle)))
    top_shift = np.tan(angle)*(wire_width/2)
    tunnel_length = wire_width
    tunnel_width = wire_width

    total_length = triangle_length + 2 * tunnel_length + 2 * gap - top_shift
    total_width = triangle_width

    centers = np.array(
        [
            [0, triangle_length - top_shift + gap + tunnel_length/2],
            [- triangle_width/4, tunnel_length/2],
            [triangle_width/4, tunnel_length/2]
        ]
    )

    back_gate = np.array(
        [
            [-triangle_width/2, 0],
            [triangle_width/2, 0],
            [wire_width/2, triangle_length-top_shift],
            [-wire_width/2, triangle_length-top_shift]
        ]
    )
    back_gate[:, 1] += tunnel_length + gap
    centers[0, 1] += tunnel_length + gap

    screening_length = tunnel_length
    middle_width = triangle_width/2 - wire_width - 2 * gap
    extra_width = triangle_width/2 - (tunnel_width+2*gap+middle_width/2)

    screening_centers = np.array(
        [
            [0, tunnel_length/2],
            [triangle_width/4 + tunnel_width/2 + gap + extra_width/2, tunnel_length/2],
            [-(triangle_width/4 + tunnel_width/2 + gap + extra_width/2), tunnel_length/2],
            [triangle_width/2 + extra_width/2 + gap, total_length/2],
            [- triangle_width/2 - extra_width/2 - gap, total_length/2]
        ]
    )

    screening_gate_right = np.array(
        [
            [total_width/2, tunnel_length + (1 + np.sqrt(2)) * gap],
            [total_width/2, total_length],
            [tunnel_width/2 + gap, total_length],
            [tunnel_width/2 + gap, total_length - tunnel_length - gap],
            #[total_width/2, tunnel_length + gap],
            #[triangle_width/2 + gap, tunnel_length],
            #[triangle_width/4 + tunnel_length/2 + gap, tunnel_length],
            #[triangle_width/4 + tunnel_length/2 + gap, 0]
        ]
    )
    
    screening_gate_left = np.copy(screening_gate_right)
    screening_gate_left[:, 0] *=  -1

    back_gates = dict()
    back_gates['back_gate'] = back_gate

    tunnel_gates = dict()
    
    tunnel_gates['top_tunnel_gate'] = rectangular_gate(center=centers[0],
                                      length=tunnel_length, 
                                      width=tunnel_width)
    tunnel_gates['left_tunnel_gate'] = rectangular_gate(center=centers[1],
                                       length=tunnel_length,
                                       width=tunnel_width)
    tunnel_gates['right_tunnel_gate'] = rectangular_gate(center=centers[2],
                                        length=tunnel_length,
                                        width=tunnel_width)
    
    screening_gates = {'left_triangle_screen_gate': screening_gate_left,
                       'right_triangle_screen_gate': screening_gate_right,
                       'low_top_screen_gate': rectangular_gate(center=screening_centers[0],
                                                                   length=screening_length,
                                                                   width=middle_width),
                       'low_left_screen_gate': rectangular_gate(center=screening_centers[1],
                                                                   length=screening_length,
                                                                   width=extra_width),
                       'low_right_screen_gate': rectangular_gate(center=screening_centers[2],
                                                                   length=screening_length,
                                                                   width=extra_width),
                       'left_screen_gate': rectangular_gate(center=screening_centers[3],
                                                           length=total_length,
                                                           width=extra_width),
                       'right_screen_gate': rectangular_gate(center=screening_centers[4],
                                                           length=total_length,
                                                           width=extra_width)
    }

    gates = {'back_gates': back_gates,
             'screening_gates': screening_gates,
             'tunnel_gates': tunnel_gates}

    return gates