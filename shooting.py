import numpy as np
from fenics import Function, FunctionSpace, interpolate, project
from solve_problem import solve_problem
from coupling import solid_to_fluid, fluid_to_solid
from scipy.sparse.linalg import LinearOperator, gmres
from parameters import Parameters
from spaces import Space
from initial import Initial
from time_structure import MacroTimeStep

# Define shooting function
def shooting_function(
    displacement_fluid: Initial,
    velocity_fluid: Initial,
    displacement_solid: Initial,
    velocity_solid: Initial,
    first_time_step,
    fluid: Space,
    solid: Space,
    interface: Space,
    param: Parameters,
    fluid_macrotimestep: MacroTimeStep,
    solid_macrotimestep: MacroTimeStep,
    adjoint,
):

    # Save old values
    displacement_solid_new = Function(solid.function_space_split[0])
    displacement_solid_new.assign(displacement_solid.new)
    velocity_solid_new = Function(solid.function_space_split[1])
    velocity_solid_new.assign(velocity_solid.new)

    # Perform one iteration
    solve_problem(
        displacement_fluid,
        velocity_fluid,
        displacement_solid,
        velocity_solid,
        fluid,
        solid,
        solid_to_fluid,
        first_time_step,
        param,
        fluid_macrotimestep,
        adjoint,
    )
    solve_problem(
        displacement_solid,
        velocity_solid,
        displacement_fluid,
        velocity_fluid,
        solid,
        fluid,
        fluid_to_solid,
        first_time_step,
        param,
        solid_macrotimestep,
        adjoint,
    )

    # Define shooting function
    shooting_function_1 = interpolate(
        project(
            displacement_solid_new - displacement_solid.new,
            solid.function_space_split[0],
        ),
        interface.function_space_split[0],
    )
    shooting_function_2 = interpolate(
        project(
            velocity_solid_new - velocity_solid.new,
            solid.function_space_split[1],
        ),
        interface.function_space_split[1],
    )

    # Represent shooting function as an array
    shooting_function_1_array = shooting_function_1.vector().get_local()
    shooting_function_2_array = shooting_function_2.vector().get_local()
    shooting_function_array = np.concatenate(
        (shooting_function_1_array, shooting_function_2_array), axis=None
    )

    return shooting_function_array


# Define linear operator for linear solver in shooting method
def shooting_newton(
    displacement_fluid: Initial,
    velocity_fluid: Initial,
    displacement_solid: Initial,
    velocity_solid: Initial,
    first_time_step,
    fluid: Space,
    solid: Space,
    interface: Space,
    param: Parameters,
    fluid_macrotimestep: MacroTimeStep,
    solid_macrotimestep: MacroTimeStep,
    adjoint,
    displacement_interface,
    velocity_interface,
    shooting_function_value,
):
    def shooting_gmres(direction):

        # Define empty functions on interface
        increment = Function(interface.function_space)
        (increment_displacement, increment_velocity) = increment.split(
            increment
        )

        # Split entrance vectors
        direction_split = np.split(direction, 2)

        # Set values of functions on interface
        increment_displacement.vector().set_local(
            displacement_interface + param.EPSILON * direction_split[0]
        )
        increment_velocity.vector().set_local(
            velocity_interface + param.EPSILON * direction_split[1]
        )

        # Interpolate functions on solid subdomain
        increment_displacement_solid = interpolate(
            increment_displacement, solid.function_space_split[0]
        )
        increment_velocity_solid = interpolate(
            increment_velocity, solid.function_space_split[1]
        )
        displacement_solid.new.assign(increment_displacement_solid)
        velocity_solid.new.assign(increment_velocity_solid)

        # Compute shooting function
        shooting_function_increment = shooting_function(
            displacement_fluid,
            velocity_fluid,
            displacement_solid,
            velocity_solid,
            first_time_step,
            fluid,
            solid,
            interface,
            param,
            fluid_macrotimestep,
            solid_macrotimestep,
            adjoint,
        )

        return (
            shooting_function_increment - shooting_function_value
        ) / param.EPSILON

    return shooting_gmres


def shooting(
    displacement_fluid: Initial,
    velocity_fluid: Initial,
    displacement_solid: Initial,
    velocity_solid: Initial,
    first_time_step,
    fluid: Space,
    solid: Space,
    interface: Space,
    param: Parameters,
    fluid_macrotimestep: MacroTimeStep,
    solid_macrotimestep: MacroTimeStep,
    adjoint,
):

    # Define initial values for Newton's method
    displacement_solid_new = Function(solid.function_space_split[0])
    velocity_solid_new = Function(solid.function_space_split[1])
    displacement_solid_new.assign(displacement_solid.old)
    velocity_solid_new.assign(velocity_solid.old)
    number_of_iterations = 0
    number_of_linear_systems = 0
    stop = False

    # Define Newton's method
    while not stop:

        number_of_iterations += 1
        number_of_linear_systems += 1
        print(f"Current iteration of Newton's method: {number_of_iterations}")

        # Define right hand side
        displacement_solid.new.assign(displacement_solid_new)
        velocity_solid.new.assign(velocity_solid_new)
        shooting_function_value = shooting_function(
            displacement_fluid,
            velocity_fluid,
            displacement_solid,
            velocity_solid,
            first_time_step,
            fluid,
            solid,
            interface,
            param,
            fluid_macrotimestep,
            solid_macrotimestep,
            adjoint,
        )
        shooting_function_value_linf = np.max(np.abs(shooting_function_value))
        if number_of_iterations == 1:

            if shooting_function_value_linf != 0.0:
                shooting_function_value_initial_linf = (
                    shooting_function_value_linf
                )
            else:
                shooting_function_value_initial_linf = 1.0

        print(
            f"Absolute error on the interface: "
            f"{shooting_function_value_linf}"
        )
        print(
            f"Relative error on the interface: "
            f"{shooting_function_value_linf / shooting_function_value_initial_linf}"
        )

        # Define linear operator
        displacement_solid_interface = interpolate(
            displacement_solid_new, interface.function_space_split[0]
        )
        velocity_solid_interface = interpolate(
            velocity_solid_new, interface.function_space_split[1]
        )
        displacement_solid_interface_array = (
            displacement_solid_interface.vector().get_local()
        )
        velocity_solid_interface_array = (
            velocity_solid_interface.vector().get_local()
        )
        linear_operator_newton = shooting_newton(
            displacement_fluid,
            velocity_fluid,
            displacement_solid,
            velocity_solid,
            first_time_step,
            fluid,
            solid,
            interface,
            param,
            fluid_macrotimestep,
            solid_macrotimestep,
            adjoint,
            displacement_solid_interface_array,
            velocity_solid_interface_array,
            shooting_function_value,
        )
        shooting_gmres = LinearOperator(
            (
                2 * param.NUMBER_ELEMENTS_HORIZONTAL + 2,
                2 * param.NUMBER_ELEMENTS_HORIZONTAL + 2,
            ),
            matvec=linear_operator_newton,
        )

        # Solve linear system
        number_of_iterations_gmres = 0

        def callback(vector):

            nonlocal number_of_iterations_gmres
            global residual_norm_gmres
            number_of_iterations_gmres += 1
            print(
                f"Current iteration of GMRES method: {number_of_iterations_gmres}"
            )
            residual_norm_gmres = np.linalg.norm(vector)

        if not adjoint:
            param.TOLERANCE_GMRES = max(
                shooting_function_value_linf, param.ABSOLUTE_TOLERANCE_NEWTON
            )
            param.EPSILON = shooting_function_value_linf
        direction, exit_code = gmres(
            shooting_gmres,
            -shooting_function_value,
            tol=param.TOLERANCE_GMRES,
            maxiter=param.MAX_ITERATIONS_GMRES,
            callback=callback,
        )
        number_of_linear_systems += number_of_iterations_gmres
        if exit_code == 0:

            print(
                f"GMRES method converged successfully after "
                f"{number_of_iterations_gmres} iterations"
            )

        else:

            print("GMRES method failed to converge.")
            print(f"Norm of residual: {residual_norm_gmres}")

        # Advance solution
        direction_split = np.split(direction, 2)
        displacement_solid_interface_array += direction_split[0]
        velocity_solid_interface_array += direction_split[1]
        displacement_solid_interface.vector().set_local(
            displacement_solid_interface_array
        )
        velocity_solid_interface.vector().set_local(
            velocity_solid_interface_array
        )
        displacement_solid_new.assign(
            interpolate(
                displacement_solid_interface, solid.function_space_split[0]
            )
        )
        velocity_solid_new.assign(
            interpolate(
                velocity_solid_interface, solid.function_space_split[1]
            )
        )

        # Check stop conditions
        if (
            shooting_function_value_linf < param.ABSOLUTE_TOLERANCE_NEWTON
            or shooting_function_value_linf
            / shooting_function_value_initial_linf
            < param.RELATIVE_TOLERANCE_NEWTON
        ):
            print(
                f"Newton's method converged successfully after "
                f"{number_of_iterations} iterations and solving "
                f"{number_of_linear_systems} linear systems."
            )
            stop = True

        elif number_of_iterations == param.MAX_ITERATIONS_NEWTON:

            print("Newton's method failed to converge.")
            stop = True
            number_of_linear_systems = -1

    displacement_fluid.iterations.append(number_of_linear_systems)
    velocity_fluid.iterations.append(number_of_linear_systems)
    displacement_solid.iterations.append(number_of_linear_systems)
    displacement_solid.iterations.append(number_of_linear_systems)

    return
