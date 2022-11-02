from fenics import project, assemble, dot, grad
from forms import (
    external_force_fluid,
    external_force_solid,
    form_fluid,
    form_solid,
    form_fluid_adjoint,
    form_solid_adjoint,
    characteristic_function_fluid,
    characteristic_function_solid,
    gauss,
)
from parameters import Parameters
from spaces import Space
from time_structure import MicroTimeStep, TimeLine

# Copy a list to the same space and in the same direction
def copy_list_forward(timeline: TimeLine, function_name):

    array = []
    macrotimestep = timeline.head
    global_size = timeline.size
    for n in range(global_size):

        microtimestep = macrotimestep.head
        local_size = macrotimestep.size
        for m in range(local_size):

            function = microtimestep.functions[function_name]
            if not microtimestep.after is None:

                array.append(function.copy(deepcopy=True))

            if (microtimestep.after is None) and (macrotimestep.after is None):

                array.append(function.copy(deepcopy=True))

            microtimestep = microtimestep.after

        macrotimestep = macrotimestep.after

    return array


# Copy a list to the same space and in the opposite direction
def copy_list_backward(timeline: TimeLine, function_name):

    array = []
    macrotimestep = timeline.tail
    global_size = timeline.size
    for n in range(global_size):

        microtimestep = macrotimestep.tail
        local_size = macrotimestep.size
        for m in range(local_size):

            function = microtimestep.functions[function_name]
            if not microtimestep.before is None:

                array.append(function.copy(deepcopy=True))

            if (microtimestep.before is None) and (
                macrotimestep.before is None
            ):

                array.append(function.copy(deepcopy=True))

            microtimestep = microtimestep.before

        macrotimestep = macrotimestep.before

    return array


# Extrapolate a list to a different space in the same direction
def extrapolate_list_forward(
    space: Space,
    space_timeline: TimeLine,
    space_interface: Space,
    space_interface_timeline: TimeLine,
    function_name,
    param: Parameters,
    transfer_function,
    subspace_index,
):
    array = []
    space_macrotimestep = space_timeline.head
    space_interface_macrotimestep = space_interface_timeline.head
    function = space_interface_macrotimestep.head.functions[function_name]
    array.append(
        transfer_function(
            function, space, space_interface, param, subspace_index
        )
    )
    global_size = space_timeline.size
    for n in range(global_size):

        space_microtimestep = space_macrotimestep.head
        local_size = space_macrotimestep.size - 1
        for m in range(local_size):

            extrapolation_proportion = (
                space_macrotimestep.tail.point
                - space_microtimestep.after.point
            ) / space_macrotimestep.dt
            function_old = space_interface_macrotimestep.head.functions[
                function_name
            ]
            function_new = space_interface_macrotimestep.tail.functions[
                function_name
            ]
            array.append(
                project(
                    extrapolation_proportion
                    * transfer_function(
                        function_old,
                        space,
                        space_interface,
                        param,
                        subspace_index,
                    )
                    + (1.0 - extrapolation_proportion)
                    * transfer_function(
                        function_new,
                        space,
                        space_interface,
                        param,
                        subspace_index,
                    ),
                    space.function_space_split[subspace_index],
                )
            )
            space_microtimestep = space_microtimestep.after

        space_macrotimestep = space_macrotimestep.after
        space_interface_macrotimestep = space_interface_macrotimestep.after

    return array


# Extrapolate a list to a different space in the same direction
def extrapolate_list_backward(
    space: Space,
    space_timeline: TimeLine,
    space_interface: Space,
    space_interface_timeline: TimeLine,
    function_name,
    param: Parameters,
    transfer_function,
    subspace_index,
):
    array = []
    space_macrotimestep = space_timeline.head
    space_interface_macrotimestep = space_interface_timeline.tail
    function = space_interface_macrotimestep.tail.functions[function_name]
    array.append(
        transfer_function(
            function, space, space_interface, param, subspace_index
        )
    )
    global_size = space_timeline.size
    for n in range(global_size):

        space_microtimestep = space_macrotimestep.head
        local_size = space_macrotimestep.size - 1
        for m in range(local_size):

            extrapolation_proportion = (
                space_macrotimestep.tail.point
                - space_microtimestep.after.point
            ) / space_macrotimestep.dt
            function_old = space_interface_macrotimestep.tail.functions[
                function_name
            ]
            function_new = space_interface_macrotimestep.head.functions[
                function_name
            ]
            array.append(
                project(
                    extrapolation_proportion
                    * transfer_function(
                        function_old,
                        space,
                        space_interface,
                        param,
                        subspace_index,
                    )
                    + (1.0 - extrapolation_proportion)
                    * transfer_function(
                        function_new,
                        space,
                        space_interface,
                        param,
                        subspace_index,
                    ),
                    space.function_space_split[subspace_index],
                )
            )
            space_microtimestep = space_microtimestep.after

        space_macrotimestep = space_macrotimestep.after
        space_interface_macrotimestep = space_interface_macrotimestep.before

    return array


# Define linear extrapolation
def linear_extrapolation(array, m, time, microtimestep: MicroTimeStep):

    time_step = microtimestep.before.dt
    point = microtimestep.point

    return (array[m] - array[m - 1]) / time_step * time + (
        array[m - 1] * point - array[m] * (point - time_step)
    ) / time_step


# Define reconstruction of the primal problem
def primal_reconstruction(array, m, time, microtimestep: MicroTimeStep):

    time_step = microtimestep.before.dt
    point = microtimestep.point
    a = (array[m + 1] - 2 * array[m] + array[m - 1]) / (
        2.0 * time_step * time_step
    )
    b = (
        (time_step - 2.0 * point) * array[m + 1]
        + 4 * point * array[m]
        + (-time_step - 2.0 * point) * array[m - 1]
    ) / (2.0 * time_step * time_step)
    c = (
        (-time_step * point + point * point) * array[m + 1]
        + (2.0 * time_step * time_step - 2.0 * point * point) * array[m]
        + (time_step * point + point * point) * array[m - 1]
    ) / (2.0 * time_step * time_step)

    return a * time * time + b * time + c


def primal_derivative(array, m, time, microtimestep: MicroTimeStep):

    time_step = microtimestep.before.dt
    point = microtimestep.point
    a = (array[m + 1] - 2 * array[m] + array[m - 1]) / (
        2.0 * time_step * time_step
    )
    b = (
        (time_step - 2.0 * point) * array[m + 1]
        + 4 * point * array[m]
        + (-time_step - 2.0 * point) * array[m - 1]
    ) / (2.0 * time_step * time_step)

    return 2.0 * a * time + b


# Define reconstruction of the adjoint problem
def adjoint_reconstruction(array, m, time, microtimestep, macrotimestep):

    size = len(array) - 1
    if m == 1 or m == size:

        return array[m]

    else:

        if microtimestep.before.before is None:

            t_average_before = 0.5 * (
                microtimestep.before.point
                + macrotimestep.microtimestep_before.point
            )

        else:

            t_average_before = 0.5 * (
                microtimestep.before.point + microtimestep.before.before.point
            )

        if microtimestep.after is None:

            t_average_after = 0.5 * (
                microtimestep.point + macrotimestep.microtimestep_after.point
            )

        else:

            t_average_after = 0.5 * (
                microtimestep.point + microtimestep.after.point
            )

        return (time - t_average_before) / (
            t_average_after - t_average_before
        ) * array[m + 1] + (time - t_average_after) / (
            t_average_before - t_average_after
        ) * array[
            m - 1
        ]


# Compute goal functionals
def goal_functional_fluid(
    displacement_fluid_array,
    velocity_fluid_array,
    fluid: Space,
    fluid_timeline: TimeLine,
    param: Parameters,
):

    global_result = 0.0
    macrotimestep = fluid_timeline.head
    global_size = fluid_timeline.size
    m = 1
    for n in range(global_size):

        microtimestep = macrotimestep.head.after
        local_size = macrotimestep.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            result = 0.0
            time_step = microtimestep.before.dt
            gauss_1, gauss_2 = gauss(microtimestep)
            result += (
                0.5
                * time_step
                * param.NU
                * characteristic_function_fluid(param)
                * dot(
                    grad(
                        linear_extrapolation(
                            velocity_fluid_array, m, gauss_1, microtimestep
                        )
                    ),
                    grad(
                        linear_extrapolation(
                            velocity_fluid_array, m, gauss_1, microtimestep
                        )
                    ),
                )
                * fluid.dx
            )
            result += (
                0.5
                * time_step
                * param.NU
                * characteristic_function_fluid(param)
                * dot(
                    grad(
                        linear_extrapolation(
                            velocity_fluid_array, m, gauss_2, microtimestep
                        )
                    ),
                    grad(
                        linear_extrapolation(
                            velocity_fluid_array, m, gauss_2, microtimestep
                        )
                    ),
                )
                * fluid.dx
            )

            m += 1
            microtimestep = microtimestep.after
            global_result += assemble(result)

        macrotimestep = macrotimestep.after

    return global_result


def goal_functional_solid(
    displacement_solid_array,
    velocity_solid_array,
    solid,
    solid_timeline,
    param,
):

    global_result = 0.0
    macrotimestep = solid_timeline.head
    global_size = solid_timeline.size
    m = 1
    for n in range(global_size):

        microtimestep = macrotimestep.head.after
        local_size = macrotimestep.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            result = 0.0
            time_step = microtimestep.before.dt
            gauss_1, gauss_2 = gauss(microtimestep)
            result += (
                0.5
                * time_step
                * param.ZETA
                * characteristic_function_solid(param)
                * dot(
                    grad(
                        linear_extrapolation(
                            displacement_solid_array, m, gauss_1, microtimestep
                        )
                    ),
                    grad(
                        linear_extrapolation(
                            displacement_solid_array, m, gauss_1, microtimestep
                        )
                    ),
                )
                * solid.dx
            )
            result += (
                0.5
                * time_step
                * param.ZETA
                * characteristic_function_solid(param)
                * dot(
                    grad(
                        linear_extrapolation(
                            displacement_solid_array, m, gauss_2, microtimestep
                        )
                    ),
                    grad(
                        linear_extrapolation(
                            displacement_solid_array, m, gauss_2, microtimestep
                        )
                    ),
                )
                * solid.dx
            )

            m += 1
            microtimestep = microtimestep.after
            global_result += assemble(result)

        macrotimestep = macrotimestep.after

    return global_result


# Compute primal residual of the fluid subproblem
def primal_residual_fluid(
    displacement_fluid_array,
    velocity_fluid_array,
    displacement_solid_array,
    velocity_solid_array,
    displacement_fluid_adjoint_array,
    velocity_fluid_adjoint_array,
    fluid: Space,
    fluid_timeline: TimeLine,
    param: Parameters,
):

    residuals = []
    macrotimestep = fluid_timeline.head
    global_size = fluid_timeline.size
    m = 1
    for n in range(global_size):

        microtimestep = macrotimestep.head.after
        local_size = macrotimestep.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            lhs = 0.0
            rhs = 0.0
            time_step = microtimestep.before.dt
            gauss_1, gauss_2 = gauss(microtimestep)
            lhs += (
                0.5
                * time_step
                * (
                    (velocity_fluid_array[m] - velocity_fluid_array[m - 1])
                    / time_step
                    * (
                        adjoint_reconstruction(
                            displacement_fluid_adjoint_array,
                            m,
                            gauss_1,
                            microtimestep,
                            macrotimestep,
                        )
                        - displacement_fluid_adjoint_array[m]
                    )
                )
                * fluid.dx
            )
            lhs += (
                0.5
                * time_step
                * (
                    (velocity_fluid_array[m] - velocity_fluid_array[m - 1])
                    / time_step
                    * (
                        adjoint_reconstruction(
                            displacement_fluid_adjoint_array,
                            m,
                            gauss_2,
                            microtimestep,
                            macrotimestep,
                        )
                        - displacement_fluid_adjoint_array[m]
                    )
                )
                * fluid.dx
            )
            lhs += (
                0.5
                * time_step
                * form_fluid(
                    linear_extrapolation(
                        displacement_fluid_array, m, gauss_1, microtimestep
                    ),
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_1, microtimestep
                    ),
                    adjoint_reconstruction(
                        displacement_fluid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep,
                        macrotimestep,
                    )
                    - displacement_fluid_adjoint_array[m],
                    adjoint_reconstruction(
                        velocity_fluid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep,
                        macrotimestep,
                    )
                    - velocity_fluid_adjoint_array[m],
                    fluid,
                    param,
                )
            )
            lhs += (
                0.5
                * time_step
                * form_fluid(
                    linear_extrapolation(
                        displacement_fluid_array, m, gauss_2, microtimestep
                    ),
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_2, microtimestep
                    ),
                    adjoint_reconstruction(
                        displacement_fluid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep,
                        macrotimestep,
                    )
                    - displacement_fluid_adjoint_array[m],
                    adjoint_reconstruction(
                        velocity_fluid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep,
                        macrotimestep,
                    )
                    - velocity_fluid_adjoint_array[m],
                    fluid,
                    param,
                )
            )
            lhs -= (
                0.5
                * time_step
                * param.GAMMA
                / fluid.cell_size
                * linear_extrapolation(
                    displacement_solid_array, m, gauss_1, microtimestep
                )
                * (
                    adjoint_reconstruction(
                        velocity_fluid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep,
                        macrotimestep,
                    )
                    - velocity_fluid_adjoint_array[m]
                )
                * fluid.ds(1)
            )
            lhs -= (
                0.5
                * time_step
                * param.GAMMA
                / fluid.cell_size
                * linear_extrapolation(
                    displacement_solid_array, m, gauss_2, microtimestep
                )
                * (
                    adjoint_reconstruction(
                        velocity_fluid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep,
                        macrotimestep,
                    )
                    - velocity_fluid_adjoint_array[m]
                )
                * fluid.ds(1)
            )
            lhs -= (
                0.5
                * time_step
                * param.NU
                * param.GAMMA
                / fluid.cell_size
                * linear_extrapolation(
                    velocity_solid_array, m, gauss_1, microtimestep
                )
                * (
                    adjoint_reconstruction(
                        displacement_fluid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep,
                        macrotimestep,
                    )
                    - displacement_fluid_adjoint_array[m]
                )
                * fluid.ds(1)
            )
            lhs -= (
                0.5
                * time_step
                * param.NU
                * param.GAMMA
                / fluid.cell_size
                * linear_extrapolation(
                    velocity_solid_array, m, gauss_2, microtimestep
                )
                * (
                    adjoint_reconstruction(
                        displacement_fluid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep,
                        macrotimestep,
                    )
                    - displacement_fluid_adjoint_array[m]
                )
                * fluid.ds(1)
            )
            rhs += (
                0.5
                * param.GOAL_FUNCTIONAL_FLUID
                * time_step
                * external_force_fluid(gauss_1, param)
                * (
                    adjoint_reconstruction(
                        displacement_fluid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep,
                        macrotimestep,
                    )
                    - displacement_fluid_adjoint_array[m]
                )
                * fluid.dx
            )
            rhs += (
                0.5
                * param.GOAL_FUNCTIONAL_FLUID
                * time_step
                * external_force_fluid(gauss_2, param)
                * (
                    adjoint_reconstruction(
                        displacement_fluid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep,
                        macrotimestep,
                    )
                    - displacement_fluid_adjoint_array[m]
                )
                * fluid.dx
            )

            m += 1
            microtimestep = microtimestep.after
            residuals.append(assemble(0.5 * rhs - 0.5 * lhs))

        macrotimestep = macrotimestep.after

    return residuals


# Compute primal residual of the solid subproblem
def primal_residual_solid(
    displacement_solid_array,
    velocity_solid_array,
    displacement_fluid_array,
    velocity_fluid_array,
    displacement_solid_adjoint_array,
    velocity_solid_adjoint_array,
    solid: Space,
    solid_timeline: TimeLine,
    param: Parameters,
):

    residuals = []
    macrotimestep = solid_timeline.head
    global_size = solid_timeline.size
    m = 1
    for n in range(global_size):

        microtimestep = macrotimestep.head.after
        local_size = macrotimestep.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            lhs = 0.0
            rhs = 0.0
            time_step = microtimestep.before.dt
            gauss_1, gauss_2 = gauss(microtimestep)
            lhs += (
                0.5
                * time_step
                * (
                    (velocity_solid_array[m] - velocity_solid_array[m - 1])
                    / time_step
                    * (
                        adjoint_reconstruction(
                            displacement_solid_adjoint_array,
                            m,
                            gauss_1,
                            microtimestep,
                            macrotimestep,
                        )
                        - displacement_solid_adjoint_array[m]
                    )
                )
                * solid.dx
            )
            lhs += (
                0.5
                * time_step
                * (
                    (velocity_solid_array[m] - velocity_solid_array[m - 1])
                    / time_step
                    * (
                        adjoint_reconstruction(
                            displacement_solid_adjoint_array,
                            m,
                            gauss_2,
                            microtimestep,
                            macrotimestep,
                        )
                        - displacement_solid_adjoint_array[m]
                    )
                )
                * solid.dx
            )
            lhs += (
                0.5
                * time_step
                * (
                    (
                        displacement_solid_array[m]
                        - displacement_solid_array[m - 1]
                    )
                    / time_step
                    * (
                        adjoint_reconstruction(
                            velocity_solid_adjoint_array,
                            m,
                            gauss_1,
                            microtimestep,
                            macrotimestep,
                        )
                        - velocity_solid_adjoint_array[m]
                    )
                )
                * solid.dx
            )
            lhs += (
                0.5
                * time_step
                * (
                    (
                        displacement_solid_array[m]
                        - displacement_solid_array[m - 1]
                    )
                    / time_step
                    * (
                        adjoint_reconstruction(
                            velocity_solid_adjoint_array,
                            m,
                            gauss_2,
                            microtimestep,
                            macrotimestep,
                        )
                        - velocity_solid_adjoint_array[m]
                    )
                )
                * solid.dx
            )
            lhs += (
                0.5
                * time_step
                * form_solid(
                    linear_extrapolation(
                        displacement_solid_array, m, gauss_1, microtimestep
                    ),
                    linear_extrapolation(
                        velocity_solid_array, m, gauss_1, microtimestep
                    ),
                    adjoint_reconstruction(
                        displacement_solid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep,
                        macrotimestep,
                    )
                    - displacement_solid_adjoint_array[m],
                    adjoint_reconstruction(
                        velocity_solid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep,
                        macrotimestep,
                    )
                    - velocity_solid_adjoint_array[m],
                    solid,
                    param,
                )
            )
            lhs += (
                0.5
                * time_step
                * form_solid(
                    linear_extrapolation(
                        displacement_solid_array, m, gauss_2, microtimestep
                    ),
                    linear_extrapolation(
                        velocity_solid_array, m, gauss_2, microtimestep
                    ),
                    adjoint_reconstruction(
                        displacement_solid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep,
                        macrotimestep,
                    )
                    - displacement_solid_adjoint_array[m],
                    adjoint_reconstruction(
                        velocity_solid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep,
                        macrotimestep,
                    )
                    - velocity_solid_adjoint_array[m],
                    solid,
                    param,
                )
            )
            lhs += (
                0.5
                * time_step
                * param.NU
                * dot(
                    grad(
                        linear_extrapolation(
                            velocity_fluid_array, m, gauss_1, microtimestep
                        )
                    ),
                    solid.normal_vector,
                )
                * (
                    adjoint_reconstruction(
                        displacement_solid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep,
                        macrotimestep,
                    )
                    - displacement_solid_adjoint_array[m]
                )
                * solid.ds(1)
            )
            lhs += (
                0.5
                * time_step
                * param.NU
                * dot(
                    grad(
                        linear_extrapolation(
                            velocity_fluid_array, m, gauss_2, microtimestep
                        )
                    ),
                    solid.normal_vector,
                )
                * (
                    adjoint_reconstruction(
                        displacement_solid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep,
                        macrotimestep,
                    )
                    - displacement_solid_adjoint_array[m]
                )
                * solid.ds(1)
            )
            rhs += (
                0.5
                * param.GOAL_FUNCTIONAL_SOLID
                * time_step
                * external_force_solid(gauss_1, param)
                * (
                    adjoint_reconstruction(
                        displacement_solid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep,
                        macrotimestep,
                    )
                    - displacement_solid_adjoint_array[m]
                )
                * solid.dx
            )
            rhs += (
                0.5
                * param.GOAL_FUNCTIONAL_SOLID
                * time_step
                * external_force_solid(gauss_2, param)
                * (
                    adjoint_reconstruction(
                        displacement_solid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep,
                        macrotimestep,
                    )
                    - displacement_solid_adjoint_array[m]
                )
                * solid.dx
            )

            m += 1
            microtimestep = microtimestep.after
            residuals.append(assemble(0.5 * rhs - 0.5 * lhs))

        macrotimestep = macrotimestep.after

    return residuals


# Compute adjoint residual of the fluid subproblem
def adjoint_residual_fluid(
    displacement_fluid_array,
    velocity_fluid_array,
    displacement_fluid_adjoint,
    velocity_fluid_adjoint,
    displacement_solid_adjoint,
    velocity_solid_adjoint,
    fluid: Space,
    fluid_timeline: TimeLine,
    param: Parameters,
):
    residuals = []
    left = True
    macrotimestep = fluid_timeline.head
    global_size = fluid_timeline.size
    m = 1
    for n in range(global_size):

        microtimestep = macrotimestep.head.after
        local_size = macrotimestep.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            lhs = 0.0
            rhs = 0.0
            time_step = microtimestep.before.dt
            gauss_1, gauss_2 = gauss(microtimestep)
            if left:

                l = m
                microtimestep_adjust = microtimestep
                left = False

            else:

                l = m - 1
                if microtimestep.before.before is None:

                    microtimestep_adjust = (
                        macrotimestep.microtimestep_before.after
                    )

                else:

                    microtimestep_adjust = microtimestep.before
                left = True

            lhs += (
                0.5
                * time_step
                * (
                    (
                        primal_derivative(
                            velocity_fluid_array,
                            l,
                            gauss_1,
                            microtimestep_adjust,
                        )
                        - (
                            velocity_fluid_array[m]
                            - velocity_fluid_array[m - 1]
                        )
                        / time_step
                    )
                    * displacement_fluid_adjoint[m]
                )
                * fluid.dx
            )
            lhs += (
                0.5
                * time_step
                * (
                    (
                        primal_derivative(
                            velocity_fluid_array,
                            l,
                            gauss_2,
                            microtimestep_adjust,
                        )
                        - (
                            velocity_fluid_array[m]
                            - velocity_fluid_array[m - 1]
                        )
                        / time_step
                    )
                    * displacement_fluid_adjoint[m]
                )
                * fluid.dx
            )
            lhs += (
                0.5
                * time_step
                * form_fluid_adjoint(
                    displacement_fluid_adjoint[m],
                    velocity_fluid_adjoint[m],
                    primal_reconstruction(
                        displacement_fluid_array,
                        l,
                        gauss_1,
                        microtimestep_adjust,
                    )
                    - linear_extrapolation(
                        displacement_fluid_array, m, gauss_1, microtimestep
                    ),
                    primal_reconstruction(
                        velocity_fluid_array, l, gauss_1, microtimestep_adjust
                    )
                    - linear_extrapolation(
                        velocity_fluid_array, m, gauss_1, microtimestep
                    ),
                    fluid,
                    param,
                )
            )
            lhs += (
                0.5
                * time_step
                * form_fluid_adjoint(
                    displacement_fluid_adjoint[m],
                    velocity_fluid_adjoint[m],
                    primal_reconstruction(
                        displacement_fluid_array,
                        l,
                        gauss_2,
                        microtimestep_adjust,
                    )
                    - linear_extrapolation(
                        displacement_fluid_array, m, gauss_2, microtimestep
                    ),
                    primal_reconstruction(
                        velocity_fluid_array, l, gauss_2, microtimestep_adjust
                    )
                    - linear_extrapolation(
                        velocity_fluid_array, m, gauss_2, microtimestep
                    ),
                    fluid,
                    param,
                )
            )
            lhs += (
                0.5
                * time_step
                * param.NU
                * dot(
                    grad(
                        primal_reconstruction(
                            velocity_fluid_array,
                            l,
                            gauss_1,
                            microtimestep_adjust,
                        )
                        - linear_extrapolation(
                            velocity_fluid_array, m, gauss_1, microtimestep
                        )
                    ),
                    fluid.normal_vector,
                )
                * displacement_solid_adjoint[m]
                * fluid.ds(1)
            )
            lhs += (
                0.5
                * time_step
                * param.NU
                * dot(
                    grad(
                        primal_reconstruction(
                            velocity_fluid_array,
                            l,
                            gauss_2,
                            microtimestep_adjust,
                        )
                        - linear_extrapolation(
                            velocity_fluid_array, m, gauss_2, microtimestep
                        )
                    ),
                    fluid.normal_vector,
                )
                * displacement_solid_adjoint[m]
                * fluid.ds(1)
            )
            rhs += (
                0.5
                * time_step
                * 2.0
                * param.NU
                * characteristic_function_fluid(param)
                * dot(
                    grad(
                        linear_extrapolation(
                            velocity_fluid_array, m, gauss_1, microtimestep
                        )
                    ),
                    grad(
                        primal_reconstruction(
                            velocity_fluid_array,
                            l,
                            gauss_1,
                            microtimestep_adjust,
                        )
                        - linear_extrapolation(
                            velocity_fluid_array, m, gauss_1, microtimestep
                        )
                    ),
                )
                * fluid.dx
            )
            rhs += (
                0.5
                * time_step
                * 2.0
                * param.NU
                * characteristic_function_fluid(param)
                * dot(
                    grad(
                        linear_extrapolation(
                            velocity_fluid_array, m, gauss_2, microtimestep
                        )
                    ),
                    grad(
                        primal_reconstruction(
                            velocity_fluid_array,
                            l,
                            gauss_2,
                            microtimestep_adjust,
                        )
                        - linear_extrapolation(
                            velocity_fluid_array, m, gauss_2, microtimestep
                        )
                    ),
                )
                * fluid.dx
            )

            m += 1
            microtimestep = microtimestep.after
            residuals.append(assemble(0.5 * rhs - 0.5 * lhs))

        macrotimestep = macrotimestep.after

    return residuals


# Compute adjoint residual of the solid subproblem
def adjoint_residual_solid(
    displacement_solid,
    velocity_solid,
    displacement_solid_adjoint,
    velocity_solid_adjoint,
    displacement_fluid_adjoint,
    velocity_fluid_adjoint,
    solid: Space,
    solid_timeline: TimeLine,
    param: Parameters,
):
    residuals = []
    left = True
    macrotimestep = solid_timeline.head
    glocal_size = solid_timeline.size
    m = 1
    for n in range(glocal_size):

        microtimestep = macrotimestep.head.after
        local_size = macrotimestep.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            lhs = 0.0
            rhs = 0.0
            time_step = microtimestep.before.dt
            gauss_1, gauss_2 = gauss(microtimestep)
            if left:

                l = m
                microtimestep_adjust = microtimestep
                left = False

            else:

                l = m - 1
                if microtimestep.before.before is None:

                    microtimestep_adjust = (
                        macrotimestep.microtimestep_before.after
                    )

                else:

                    microtimestep_adjust = microtimestep.before
                left = True

            lhs += (
                0.5
                * time_step
                * (
                    (
                        primal_derivative(
                            velocity_solid, l, gauss_1, microtimestep_adjust
                        )
                        - (velocity_solid[m] - velocity_solid[m - 1])
                        / time_step
                    )
                    * displacement_solid_adjoint[m]
                )
                * solid.dx
            )
            lhs += (
                0.5
                * time_step
                * (
                    (
                        primal_derivative(
                            velocity_solid, l, gauss_2, microtimestep_adjust
                        )
                        - (velocity_solid[m] - velocity_solid[m - 1])
                        / time_step
                    )
                    * displacement_solid_adjoint[m]
                )
                * solid.dx
            )
            lhs += (
                0.5
                * time_step
                * (
                    (
                        primal_derivative(
                            displacement_solid,
                            l,
                            gauss_1,
                            microtimestep_adjust,
                        )
                        - (displacement_solid[m] - displacement_solid[m - 1])
                        / time_step
                    )
                    * velocity_solid_adjoint[m]
                )
                * solid.dx
            )
            lhs += (
                0.5
                * time_step
                * (
                    (
                        primal_derivative(
                            displacement_solid,
                            l,
                            gauss_2,
                            microtimestep_adjust,
                        )
                        - (displacement_solid[m] - displacement_solid[m - 1])
                        / time_step
                    )
                    * velocity_solid_adjoint[m]
                )
                * solid.dx
            )
            lhs += (
                0.5
                * time_step
                * form_solid_adjoint(
                    displacement_solid_adjoint[m],
                    velocity_solid_adjoint[m],
                    primal_reconstruction(
                        displacement_solid, l, gauss_1, microtimestep_adjust
                    )
                    - linear_extrapolation(
                        displacement_solid, m, gauss_1, microtimestep
                    ),
                    primal_reconstruction(
                        velocity_solid, l, gauss_1, microtimestep_adjust
                    )
                    - linear_extrapolation(
                        velocity_solid, m, gauss_1, microtimestep
                    ),
                    solid,
                    param,
                )
            )
            lhs += (
                0.5
                * time_step
                * form_solid_adjoint(
                    displacement_solid_adjoint[m],
                    velocity_solid_adjoint[m],
                    primal_reconstruction(
                        displacement_solid, l, gauss_2, microtimestep_adjust
                    )
                    - linear_extrapolation(
                        displacement_solid, m, gauss_2, microtimestep
                    ),
                    primal_reconstruction(
                        velocity_solid, l, gauss_2, microtimestep_adjust
                    )
                    - linear_extrapolation(
                        velocity_solid, m, gauss_2, microtimestep
                    ),
                    solid,
                    param,
                )
            )
            lhs -= (
                0.5
                * time_step
                * param.GAMMA
                / solid.cell_size
                * (
                    primal_reconstruction(
                        displacement_solid, l, gauss_1, microtimestep_adjust
                    )
                    - linear_extrapolation(
                        displacement_solid, m, gauss_1, microtimestep
                    )
                )
                * velocity_fluid_adjoint[m]
                * solid.ds(1)
            )
            lhs -= (
                0.5
                * time_step
                * param.GAMMA
                / solid.cell_size
                * (
                    primal_reconstruction(
                        displacement_solid, l, gauss_2, microtimestep_adjust
                    )
                    - linear_extrapolation(
                        displacement_solid, m, gauss_2, microtimestep
                    )
                )
                * velocity_fluid_adjoint[m]
                * solid.ds(1)
            )
            lhs -= (
                0.5
                * time_step
                * param.NU
                * param.GAMMA
                / solid.cell_size
                * (
                    primal_reconstruction(
                        velocity_solid, l, gauss_1, microtimestep_adjust
                    )
                    - linear_extrapolation(
                        velocity_solid, m, gauss_1, microtimestep
                    )
                )
                * displacement_fluid_adjoint[m]
                * solid.ds(1)
            )
            lhs -= (
                0.5
                * time_step
                * param.NU
                * param.GAMMA
                / solid.cell_size
                * (
                    primal_reconstruction(
                        velocity_solid, l, gauss_2, microtimestep_adjust
                    )
                    - linear_extrapolation(
                        velocity_solid, m, gauss_2, microtimestep
                    )
                )
                * displacement_fluid_adjoint[m]
                * solid.ds(1)
            )
            rhs += (
                0.5
                * time_step
                * 2.0
                * param.ZETA
                * characteristic_function_solid(param)
                * dot(
                    grad(
                        linear_extrapolation(
                            displacement_solid, m, gauss_1, microtimestep
                        )
                    ),
                    grad(
                        primal_reconstruction(
                            displacement_solid,
                            l,
                            gauss_1,
                            microtimestep_adjust,
                        )
                        - linear_extrapolation(
                            displacement_solid, m, gauss_1, microtimestep
                        )
                    ),
                )
                * solid.dx
            )
            rhs += (
                0.5
                * time_step
                * 2.0
                * param.ZETA
                * characteristic_function_solid(param)
                * dot(
                    grad(
                        linear_extrapolation(
                            displacement_solid, m, gauss_2, microtimestep
                        )
                    ),
                    grad(
                        primal_reconstruction(
                            displacement_solid,
                            l,
                            gauss_2,
                            microtimestep_adjust,
                        )
                        - linear_extrapolation(
                            displacement_solid, m, gauss_2, microtimestep
                        )
                    ),
                )
                * solid.dx
            )

            m += 1
            microtimestep = microtimestep.after
            residuals.append(assemble(0.5 * rhs - 0.5 * lhs))

        macrotimestep = macrotimestep.after

    return residuals
