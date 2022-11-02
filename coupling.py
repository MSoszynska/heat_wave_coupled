from fenics import Function, FunctionSpace, vertex_to_dof_map
from spaces import Space
from parameters import Parameters

# Copy two layers of dofs around the interface from fluid to solid
def fluid_to_solid(
    function, solid: Space, fluid: Space, param: Parameters, subspace_index
):

    function_vector = function.vector()
    vertex_to_dof_fluid = vertex_to_dof_map(
        fluid.function_space_split[subspace_index]
    )
    result = Function(solid.function_space_split[subspace_index])
    result_vector = result.vector()
    vector_to_dif_solid = vertex_to_dof_map(
        solid.function_space_split[subspace_index]
    )
    horizontal = param.NUMBER_ELEMENTS_HORIZONTAL + 1
    vertical = param.NUMBER_ELEMENTS_VERTICAL + 1
    for i in range(2):

        for j in range(horizontal):

            result_vector[
                vector_to_dif_solid[(vertical - i - 1) * horizontal + j]
            ] = function_vector[vertex_to_dof_fluid[i * horizontal + j]]

    return result


# Copy two layers of dofs around the interface from solid to fluid
def solid_to_fluid(
    function, fluid: Space, solid: Space, param: Parameters, subspace_index
):

    function_vector = function.vector()
    vertex_to_dof_solid = vertex_to_dof_map(
        solid.function_space_split[subspace_index]
    )
    result = Function(fluid.function_space_split[subspace_index])
    result_vector = result.vector()
    vertex_to_dof_fluid = vertex_to_dof_map(
        fluid.function_space_split[subspace_index]
    )
    horizontal = param.NUMBER_ELEMENTS_HORIZONTAL + 1
    vertical = param.NUMBER_ELEMENTS_VERTICAL + 1
    for i in range(2):

        for j in range(horizontal):

            result_vector[
                vertex_to_dof_fluid[i * horizontal + j]
            ] = function_vector[
                vertex_to_dof_solid[(vertical - i - 1) * horizontal + j]
            ]

    return result
