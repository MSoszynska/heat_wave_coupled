from error_estimate import (
    copy_list_forward,
    copy_list_backward,
    extrapolate_list_forward,
    extrapolate_list_backward,
    primal_residual_fluid,
    primal_residual_solid,
    adjoint_residual_fluid,
    adjoint_residual_solid,
    goal_functional_fluid,
    goal_functional_solid,
)
from coupling import fluid_to_solid, solid_to_fluid
from parameters import Parameters
from spaces import Space
from time_structure import TimeLine


def compute_residuals(
    fluid: Space,
    solid: Space,
    param: Parameters,
    fluid_timeline: TimeLine,
    solid_timeline: TimeLine,
):

    # Create text file
    residuals_txt = open("residuals.txt", "a")

    # Prepare arrays of solutions
    displacement_fluid_to_fluid_array = copy_list_forward(
        fluid_timeline, "primal_displacement"
    )
    displacement_fluid_to_solid_array = extrapolate_list_forward(
        solid,
        solid_timeline,
        fluid,
        fluid_timeline,
        "primal_displacement",
        param,
        fluid_to_solid,
        0,
    )
    velocity_fluid_to_fluid_array = copy_list_forward(
        fluid_timeline, "primal_velocity"
    )
    velocity_fluid_to_solid_array = extrapolate_list_forward(
        solid,
        solid_timeline,
        fluid,
        fluid_timeline,
        "primal_velocity",
        param,
        fluid_to_solid,
        1,
    )
    displacement_solid_to_solid_array = copy_list_forward(
        solid_timeline, "primal_displacement"
    )
    displacement_solid_to_fluid_array = extrapolate_list_forward(
        fluid,
        fluid_timeline,
        solid,
        solid_timeline,
        "primal_displacement",
        param,
        solid_to_fluid,
        0,
    )
    velocity_solid_to_solid_array = copy_list_forward(
        solid_timeline, "primal_velocity"
    )
    velocity_solid_to_fluid_array = extrapolate_list_forward(
        fluid,
        fluid_timeline,
        solid,
        solid_timeline,
        "primal_velocity",
        param,
        solid_to_fluid,
        1,
    )
    displacement_adjoint_fluid_to_fluid_array = copy_list_backward(
        fluid_timeline, "adjoint_displacement"
    )
    displacement_adjoint_fluid_to_solid_array = extrapolate_list_backward(
        solid,
        solid_timeline,
        fluid,
        fluid_timeline,
        "adjoint_displacement",
        param,
        fluid_to_solid,
        0,
    )
    velocity_adjoint_fluid_to_fluid_array = copy_list_backward(
        fluid_timeline, "adjoint_velocity"
    )
    velocity_adjoint_fluid_to_solid_array = extrapolate_list_backward(
        solid,
        solid_timeline,
        fluid,
        fluid_timeline,
        "adjoint_velocity",
        param,
        fluid_to_solid,
        1,
    )
    displacement_adjoint_solid_to_solid_array = copy_list_backward(
        solid_timeline, "adjoint_displacement"
    )
    displacement_adjoint_solid_to_fluid_array = extrapolate_list_backward(
        fluid,
        fluid_timeline,
        solid,
        solid_timeline,
        "adjoint_displacement",
        param,
        solid_to_fluid,
        0,
    )
    velocity_adjoint_solid_to_solid_array = copy_list_backward(
        solid_timeline, "adjoint_velocity"
    )
    velocity_adjoint_solid_to_fluid_array = extrapolate_list_backward(
        fluid,
        fluid_timeline,
        solid,
        solid_timeline,
        "adjoint_velocity",
        param,
        solid_to_fluid,
        1,
    )

    # Compute residuals
    primal_fluid = primal_residual_fluid(
        displacement_fluid_to_fluid_array,
        velocity_fluid_to_fluid_array,
        displacement_solid_to_fluid_array,
        velocity_solid_to_fluid_array,
        displacement_adjoint_fluid_to_fluid_array,
        velocity_adjoint_fluid_to_fluid_array,
        fluid,
        fluid_timeline,
        param,
    )
    print(f"Primal residual for the fluid subproblem: " f"{sum(primal_fluid)}")
    residuals_txt.write(
        f"Primal residual for the fluid subproblem: "
        f"{sum(primal_fluid)} \r\n"
    )
    primal_solid = primal_residual_solid(
        displacement_solid_to_solid_array,
        velocity_solid_to_solid_array,
        displacement_fluid_to_solid_array,
        velocity_fluid_to_solid_array,
        displacement_adjoint_solid_to_solid_array,
        velocity_adjoint_solid_to_solid_array,
        solid,
        solid_timeline,
        param,
    )
    print(f"Primal residual for the solid subproblem: " f"{sum(primal_solid)}")
    residuals_txt.write(
        f"Primal residual for the solid subproblem: "
        f"{sum(primal_solid)} \r\n"
    )
    adjoint_fluid = adjoint_residual_fluid(
        displacement_fluid_to_fluid_array,
        velocity_fluid_to_fluid_array,
        displacement_adjoint_fluid_to_fluid_array,
        velocity_adjoint_fluid_to_fluid_array,
        displacement_adjoint_solid_to_fluid_array,
        velocity_adjoint_solid_to_fluid_array,
        fluid,
        fluid_timeline,
        param,
    )
    print(
        f"Adjoint residual for the fluid subproblem: " f"{sum(adjoint_fluid)}"
    )
    residuals_txt.write(
        f"Adjoint residual for the fluid subproblem: "
        f"{sum(adjoint_fluid)} \r\n"
    )
    adjoint_solid = adjoint_residual_solid(
        displacement_solid_to_solid_array,
        velocity_solid_to_solid_array,
        displacement_adjoint_solid_to_solid_array,
        velocity_adjoint_solid_to_solid_array,
        displacement_adjoint_fluid_to_solid_array,
        velocity_adjoint_fluid_to_solid_array,
        solid,
        solid_timeline,
        param,
    )
    print(
        f"Adjoint residual for the solid subproblem: " f"{sum(adjoint_solid)}"
    )
    residuals_txt.write(
        f"Adjoint residual for the solid subproblem: "
        f"{sum(adjoint_solid)} \r\n"
    )

    # Compute goal functional
    if param.GOAL_FUNCTIONAL_FLUID and param.GOAL_FUNCTIONAL_SOLID:

        goal_functional = goal_functional_fluid(
            displacement_fluid_to_fluid_array,
            velocity_fluid_to_fluid_array,
            fluid,
            fluid_timeline,
            param,
        ) + goal_functional_solid(
            displacement_solid_to_solid_array,
            velocity_solid_to_solid_array,
            solid,
            solid_timeline,
            param,
        )

    elif param.GOAL_FUNCTIONAL_FLUID:

        goal_functional = goal_functional_fluid(
            displacement_fluid_to_fluid_array,
            velocity_fluid_to_fluid_array,
            fluid,
            fluid_timeline,
            param,
        )

    else:

        goal_functional = goal_functional_solid(
            displacement_solid_to_solid_array,
            velocity_solid_to_solid_array,
            solid,
            solid_timeline,
            param,
        )

    print(f"Value of goal functional: {goal_functional}")
    residuals_txt.write(f"Value of goal functional: {goal_functional} \r\n")

    residuals_txt.close()

    fluid_residual = 0
    for i in range(len(primal_fluid)):
        fluid_residual += abs(primal_fluid[i] + adjoint_fluid[i])
    print(f"Value of fluid residual: {fluid_residual}")

    solid_residual = 0
    for i in range(len(primal_solid)):
        solid_residual += abs(primal_solid[i] + adjoint_solid[i])
    print(f"Value of solid residual: {solid_residual}")

    return [
        primal_fluid,
        primal_solid,
        adjoint_fluid,
        adjoint_solid,
        goal_functional,
    ]
