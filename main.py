import os

from fenics import (
    parameters,
    RectangleMesh,
    Point,
    BoundaryMesh,
    SubMesh,
    HDF5File,
    MPI,
)
from parameters import Parameters
from spaces import Inner_boundary, Space
from time_structure import TimeLine, split
from time_stepping import time_stepping
from forms import (
    bilinear_form_fluid,
    functional_fluid,
    bilinear_form_solid,
    functional_solid,
    functional_fluid_adjoint_initial,
    functional_solid_adjoint_initial,
    bilinear_form_fluid_adjoint,
    functional_fluid_adjoint,
    bilinear_form_solid_adjoint,
    functional_solid_adjoint,
    Problem,
)
from relaxation import relaxation
from shooting import shooting
from compute_residuals import compute_residuals
from refine import refine

parameters["allow_extrapolation"] = True
param = Parameters()

# Create meshes
mesh_f = RectangleMesh(
    Point(0.0, 0.0),
    Point(4.0, 1.0),
    param.NUMBER_ELEMENTS_HORIZONTAL,
    param.NUMBER_ELEMENTS_VERTICAL,
    diagonal="right",
)
mesh_s = RectangleMesh(
    Point(0.0, -1.0),
    Point(4.0, 0.0),
    param.NUMBER_ELEMENTS_HORIZONTAL,
    param.NUMBER_ELEMENTS_VERTICAL,
    diagonal="left",
)
boundary_mesh = BoundaryMesh(mesh_f, "exterior")
inner_boundary = Inner_boundary()
mesh_i = SubMesh(boundary_mesh, inner_boundary)

# Create function spaces
fluid = Space(mesh_f, param.LOCAL_MESH_SIZE_FLUID)
solid = Space(mesh_s, param.LOCAL_MESH_SIZE_SOLID)
interface = Space(mesh_i)

# Define variational forms
fluid.primal_problem = Problem(
    bilinear_form_fluid, functional_fluid, functional_fluid
)
fluid.adjoint_problem = Problem(
    bilinear_form_fluid_adjoint,
    functional_fluid_adjoint_initial,
    functional_fluid_adjoint,
)
solid.primal_problem = Problem(
    bilinear_form_solid, functional_solid, functional_solid
)
solid.adjoint_problem = Problem(
    bilinear_form_solid_adjoint,
    functional_solid_adjoint_initial,
    functional_solid_adjoint,
)

# Create time interval structures
fluid_timeline = TimeLine()
fluid_timeline.unify(
    param.TIME_STEP, param.LOCAL_MESH_SIZE_FLUID, param.GLOBAL_MESH_SIZE
)
solid_timeline = TimeLine()
solid_timeline.unify(
    param.TIME_STEP, param.LOCAL_MESH_SIZE_SOLID, param.GLOBAL_MESH_SIZE
)

# Set deoupling method
if param.RELAXATION:

    decoupling = relaxation

else:

    decoupling = shooting

# Refine time meshes
fluid_size = fluid_timeline.size_global - fluid_timeline.size
solid_size = solid_timeline.size_global - solid_timeline.size
for i in range(param.REFINEMENT_LEVELS):

    fluid_refinements_txt = open(
        f"fluid_{fluid_size}-{solid_size}_refinements.txt", "r"
    )
    solid_refinements_txt = open(
        f"solid_{fluid_size}-{solid_size}_refinements.txt", "r"
    )
    fluid_refinements = [bool(int(x)) for x in fluid_refinements_txt.read()]
    solid_refinements = [bool(int(x)) for x in solid_refinements_txt.read()]
    fluid_refinements_txt.close()
    solid_refinements_txt.close()
    fluid_timeline.refine(fluid_refinements)
    solid_timeline.refine(solid_refinements)
    split(fluid_timeline, solid_timeline)
    fluid_size = fluid_timeline.size_global - fluid_timeline.size
    solid_size = solid_timeline.size_global - solid_timeline.size
    print(f"Global number of macro time-steps: {fluid_timeline.size}")
    print(
        f"Global number of micro time-steps in the fluid timeline: {fluid_size}"
    )
    print(
        f"Global number of micro time-steps in the solid timeline: {solid_size}"
    )
fluid_timeline.print(True)
solid_timeline.print(True)

# Create directory
fluid_size = fluid_timeline.size_global - fluid_timeline.size
solid_size = solid_timeline.size_global - solid_timeline.size
try:

    os.makedirs(f"{fluid_size}-{solid_size}")

except FileExistsError:

    pass
os.chdir(f"{fluid_size}-{solid_size}")

# Perform time-stepping of the primal problem
adjoint = False
if param.COMPUTE_PRIMAL:
    time_stepping(
        fluid,
        solid,
        interface,
        param,
        decoupling,
        fluid_timeline,
        solid_timeline,
        adjoint,
    )
fluid_timeline.load(fluid, "fluid", adjoint)
solid_timeline.load(solid, "solid", adjoint)

# Perform time-stepping of the adjoint problem
adjoint = True
if param.COMPUTE_ADJOINT:
    time_stepping(
        fluid,
        solid,
        interface,
        param,
        decoupling,
        fluid_timeline,
        solid_timeline,
        adjoint,
    )
fluid_timeline.load(fluid, "fluid", adjoint)
solid_timeline.load(solid, "solid", adjoint)

# Compute residuals
(
    primal_residual_fluid,
    primal_residual_solid,
    adjoint_residual_fluid,
    adjoint_residual_solid,
    goal_functional,
) = compute_residuals(fluid, solid, param, fluid_timeline, solid_timeline)

# Refine mesh
refine(
    primal_residual_fluid,
    primal_residual_solid,
    adjoint_residual_fluid,
    adjoint_residual_solid,
    fluid_timeline,
    solid_timeline,
)
