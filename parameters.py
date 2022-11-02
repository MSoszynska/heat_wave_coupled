from fenics import Constant

# Define parameters
class Parameters:
    def __init__(
        self,
        nu=Constant(0.001),
        beta=Constant((2.0, 0.0)),
        zeta=Constant(1000.0),
        delta=Constant(0.1),
        gamma=Constant(1.0),
        time_step=0.02,
        global_mesh_size=50,
        local_mesh_size_fluid=1,
        local_mesh_size_solid=1,
        number_elements_horizontal=80,
        number_elements_vertical=20,
        tau=Constant(0.5),
        absolute_tolerance_relaxation=1.0e-10,
        relative_tolerance_relaxation=1.0e-8,
        max_iterations_relaxation=50,
        epsilon=1.0e-10,
        absolute_tolerance_newton=1.0e-10,
        relative_tolerance_newton=1.0e-8,
        max_iterations_newton=15,
        tolerance_gmres=1.0e-8,
        max_iterations_gmres=20,
        relaxation=True,
        shooting=False,
        goal_functional_fluid=True,
        goal_functional_solid=True,
        compute_primal=False,
        compute_adjoint=False,
        refinement_levels=4,
    ):

        # Define problem parameters
        self.NU = nu
        self.BETA = beta
        self.ZETA = zeta
        self.DELTA = delta
        self.GAMMA = gamma

        # Define time step on the coarsest level
        self.TIME_STEP = time_step

        # Define number of macro time steps on the coarsest level
        self.GLOBAL_MESH_SIZE = global_mesh_size

        # Define number of micro time-steps for fluid
        self.LOCAL_MESH_SIZE_FLUID = local_mesh_size_fluid

        # Define number of micro time-steps for solid
        self.LOCAL_MESH_SIZE_SOLID = local_mesh_size_solid

        # Define number of mesh cells
        self.NUMBER_ELEMENTS_HORIZONTAL = number_elements_horizontal
        self.NUMBER_ELEMENTS_VERTICAL = number_elements_vertical

        # Define relaxation parameters
        self.TAU = tau
        self.ABSOLUTE_TOLERANCE_RELAXATION = absolute_tolerance_relaxation
        self.RELATIVE_TOLERANCE_RELAXATION = relative_tolerance_relaxation
        self.MAX_ITERATIONS_RELAXATION = max_iterations_relaxation

        # Define parameters for Newton's method
        self.EPSILON = epsilon
        self.ABSOLUTE_TOLERANCE_NEWTON = absolute_tolerance_newton
        self.RELATIVE_TOLERANCE_NEWTON = relative_tolerance_newton
        self.MAX_ITERATIONS_NEWTON = max_iterations_newton

        # Define parameters for GMRES method
        self.TOLERANCE_GMRES = tolerance_gmres
        self.MAX_ITERATIONS_GMRES = max_iterations_gmres

        # Choose decoupling method
        self.RELAXATION = relaxation
        self.SHOOTING = shooting

        # Choose goal functional
        self.GOAL_FUNCTIONAL_FLUID = goal_functional_fluid
        self.GOAL_FUNCTIONAL_SOLID = goal_functional_solid

        # Decide if primal and adjoint problems should be solved
        self.COMPUTE_PRIMAL = compute_primal
        self.COMPUTE_ADJOINT = compute_adjoint

        # Set number of refinement levels
        self.REFINEMENT_LEVELS = refinement_levels
