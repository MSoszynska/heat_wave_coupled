from fenics import (
    near,
    SubDomain,
    CellDiameter,
    FacetNormal,
    Measure,
    MeshFunction,
    FiniteElement,
    FunctionSpace,
)

# Define boundary
def boundary_up(x, on_boundary):
    return on_boundary and near(x[1], 1.0)


def boundary_down(x, on_boundary):
    return on_boundary and near(x[1], -1.0)


def boundary_right(x, on_boundary):
    return on_boundary and near(x[0], 0.0)


def boundary_left(x, on_boundary):
    return on_boundary and near(x[0], 4.0)


def boundary_between(x, on_boundary):
    return on_boundary and near(x[1], 0.0)


# Define interface
class Inner_boundary(SubDomain):
    def inside(self, x, on_boundary):

        return near(x[1], 0.0)


# Store space attributes
class Space:
    def __init__(self, mesh, N=0):

        # Define mesh parameters
        self.mesh = mesh
        self.cell_size = CellDiameter(mesh)
        self.normal_vector = FacetNormal(mesh)

        # Define measures
        inner_boundary = Inner_boundary()
        sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        sub_domains.set_all(0)
        inner_boundary.mark(sub_domains, 1)
        self.dx = Measure("dx", domain=mesh)
        self.ds = Measure("ds", domain=mesh, subdomain_data=sub_domains)

        # Define function spaces
        finite_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        self.function_space = FunctionSpace(
            mesh, finite_element * finite_element
        )
        self.function_space_split = [
            self.function_space.sub(0).collapse(),
            self.function_space.sub(1).collapse(),
        ]
