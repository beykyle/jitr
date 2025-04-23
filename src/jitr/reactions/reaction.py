from ..utils import mass, constants
import periodictable


# TODO add GS spin
class Particle:
    """
    Represents a particle with rest mass and charge.

    Attributes:
        m0 (float): The rest mass of the particle.
    """

    def __init__(self, m0, q):
        """
        Initializes a Particle instance.

        Params:
            m0 (float): The rest mass of the particle.
            q (float): The electric charge of the particle.
        """
        self.m0 = m0
        self.q = q

    def latex(self):
        """
        Returns the LaTeX representation of the particle.

        Returns:
            str: LaTeX string.
        """
        pass

    def __str__(self):
        """
        Returns the string representation of the particle.

        Returns:
            str: String representation.
        """
        return self.__repr__()

    def __eq__(self, other):
        """
        Checks equality with another particle.

        Params:
            other (Particle): Another particle to compare.

        Returns:
            bool: True if equal, False otherwise.
        """
        pass

    def __repr__(self):
        """
        Returns the symbolic representation of the particle.

        Returns:
            str: Symbolic representation.
        """
        pass

    @classmethod
    def parse(cls, p):
        if p is None:
            return None
        elif isinstance(p, tuple):
            return Nucleus(*p)
        elif isinstance(p, Nucleus):
            return p
        elif isinstance(p, Gamma):
            return p
        elif isinstance(p, str):
            return NotImplemented
            # TODO parse from str
        else:
            return ValueError(f"Can't parse a particle from a {type(p)}")


class Nucleus(Particle):
    """
    Represents a Nucleus with atomic mass number A and atomic number Z. Nucleons
    can also be represented as a Nucleus.

    Attributes:
        A (int): Atomic mass number.
        Z (int): Atomic number.
        Efn (float): Neutron Fermi energy.
        Efp (float): Proton Fermi energy.
    """

    def __init__(self, A: int, Z: int, **mass_kwargs):
        """
        Initializes a Nucleus instance.

        Params:
            A (int): Atomic mass number (must be greater than 0).
            Z (int): Atomic number (must be greater than or equal to 0).
            mass_kwargs: Additional keyword arguments for mass calculations.
        """

        self.A = A
        self.Z = Z
        if A > 0:
            m0 = mass.mass(A, Z, **mass_kwargs)[0]
        else:
            m0 = 0
        super().__init__(m0, Z)

        if A > 1:
            self.Efn = mass.neutron_fermi_energy(self.A, self.Z, **mass_kwargs)[0]
        else:
            self.Efn = 0

        if Z >= 1 and A > 1:
            self.Efp = mass.proton_fermi_energy(self.A, self.Z, **mass_kwargs)[0]
        else:
            self.Efp = 0

    def __add__(self, other):
        """
        Adds two particles together.

        Params:
            other (Nucleus): Another particle to add.

        Returns:
            Nucleus: New particle resulting from the addition.
        """
        if isinstance(other, Nucleus):
            return Nucleus(self.A + other.A, self.Z + other.Z)
        elif isinstance(other, tuple):
            return Nucleus(self.A + other[0], self.Z + other[1])
        elif isinstance(other, Gamma):
            return self
        else:
            raise ValueError(f"Cannot add {type(other)} to a Nucleus")

    def __sub__(self, other):
        """
        Subtracts one particle from another.

        Params:
            other (Nucleus): Another particle to subtract.

        Returns:
            Nucleus: New particle resulting from the subtraction.
        """
        if isinstance(other, Nucleus):
            return Nucleus(self.A - other.A, self.Z - other.Z)
        elif isinstance(other, tuple):
            return Nucleus(self.A - other[0], self.Z - other[1])
        elif isinstance(other, Gamma):
            return self
        else:
            raise ValueError(f"Cannot add {type(other)} to a Nucleus")

    def latex(self):
        """
        Returns the LaTeX representation of the particle.

        Returns:
            str: LaTeX string.
        """
        return get_latex(self.A, self.Z)

    def __eq__(self, other):
        """
        Checks equality with another particle.

        Params:
            other (Particle): Another particle to compare.

        Returns:
            bool: True if equal, False otherwise.
        """
        if isinstance(other, Nucleus):
            return self.A == other.A and self.Z == other.Z
        elif isinstance(other, tuple):
            return self.A == other[0] and self.Z == other[1]
        else:
            return False

    def __repr__(self):
        """
        Returns the symbolic representation of the particle.

        Returns:
            str: Symbolic representation.
        """
        return get_symbol(self.A, self.Z)

    def __iter__(self):
        """
        Allows unpacking of a Nucleus instance into (A, Z).

        Returns:
            iterator: An iterator over the atomic mass number and atomic number.
        """
        return iter((self.A, self.Z))


class Gamma(Particle):
    """
    Represents a gamma-ray

    Inherits from Particle with m0=0 and q=0
    """

    def __init__(self):
        """
        Initializes a Gamma object
        """
        super().__init__(0, 0)

    def latex(self):
        return r"$\gamma$"

    def __repr__(self):
        return "gamma"


class Electron(Particle):
    """
    Represents an electron

    Inherits from Particle
    """

    def __init__(self):
        """
        Initializes an Electron
        """
        super().__init__(constants.MASS_E, -1)

    def latex(self):
        return r"$e^{-}$"

    def __repr__(self):
        return "e-"


class Positron(Particle):
    """
    Represents a positron

    Inherits from Particle
    """

    def __init__(self):
        """
        Initializes an possitron
        """
        super().__init__(constants.MASS_E, +1)

    def latex(self):
        return r"$e^{+}$"

    def __repr__(self):
        return "e+:"


# TODO support multiple processes/products, e.g. (n,2NF) or (N,3N)
class Reaction:
    """Represents a nuclear reaction of the form A + a -> b + B.

    Attributes:
        target (Particle): The target particle.
        projectile (Particle): The projectile particle.
        product (Particle):
        residual (Particle or None): The residual particle, or None for 'abs'
            or 'tot' processes.
        compound_system (Particle): The compound system formed by target and
            projectile.
        process (str):
        Q (float or None): The Q-value of the reaction. If Q is None,
            that means it is unspecified for the given process.
        reaction_string (str): The string representation of the reaction.
        reaction_latex (str): The LaTeX representation of the reaction.
    """

    def __init__(
        self,
        target: Particle,
        projectile: Particle,
        product: Particle = None,
        residual: Particle = None,
        process: str = None,
        **mass_kwargs,
    ):
        """Initializes a Reaction instance.

        Args:
            target (Particle): The target particle.
            projectile (Particle): The projectile particle.
            product (Particle):
            process (str): The product particle or a string
                denoting the process. Currently supported are 'tot', 'el',
                'inl', 'abs', 'x', 'sct', 'non', and 'f'. Process types
                are defined in the EXFOR manual:
                https://www-nds.iaea.org/nrdc/nrdc_doc/bnl-ncs-063380-200105.pdf
            residual (Particle or None): The residual particle, or None for
                'abs' or 'tot' processes.

        Raises:
            ValueError: If isospin is not conserved or if invalid product/
            residual types are provided.
        """
        self.target = Particle.parse(target)
        self.projectile = Particle.parse(projectile)
        self.compound_system = self.target + self.projectile

        if product is not None:
            product = Particle.parse(product)
        if residual is not None:
            residual = Particle.parse(residual)

        # parse the process string and ensuure the reactants are consistent
        residual_in_string = True
        if process:
            self.process = process.lower()

            if self.process in ["el", "inl", "sct"]:
                if (product and product != self.projectile) or (
                    residual and residual != self.target
                ):
                    raise ValueError(
                        "Invalid scattering process reaction configuration."
                    )
                self.product = self.projectile
                self.residual = self.target
                self.Q = 0
                residual_in_string = False
            elif self.process in ["abs", "f"]:
                if product or residual != self.compound_system:
                    raise ValueError(
                        f"Invalid '{self.process}' process reaction configuration."
                        + f"\nThere should be no product but {product} was provided"
                        + f"\nResidual should be the {self.compound_system} but"
                        + f" {residual} was provided"
                    )
                self.product = None
                self.residual = self.compound_system
                if self.process == "abs":
                    self.Q = (
                        self.projectile.m0 + self.target.m0 - self.compound_system.m0
                    )
                else:
                    self.Q = None  # don't know fragments
                residual_in_string = False
            elif self.process == "tot" or self.process == "non":
                if product or residual:
                    raise ValueError(
                        f"Invalid {self.process} process reaction configuration."
                        + f"\nThere should be no product but {product} was provided"
                        + f"\nThere should be no residual but {residual} was provided"
                    )
                self.product = None
                self.residual = None
                self.Q = None
                residual_in_string = False
            elif self.process == "x":
                if product or not residual:
                    raise ValueError(
                        f"Invalid {self.process} process reaction configuration."
                        + f"\nThere should be no product but {product} was provided"
                        + f"\nThere must be a residual provided"
                    )
                self.product = None
                self.residual = Particle.parse(residual)
                self.Q = None

            self.reaction_string = (
                f"{self.target}({self.projectile}," + f"{self.process.lower()})"
            )
            self.reaction_latex = (
                f"{self.target.latex()}({self.projectile.latex()},"
                + f"{self.process.lower()})"
            )
            if residual_in_string:
                self.reaction_string += f"{self.residual}"
                self.reaction_latex += f"{self.residual.latex()}"
        else:
            if product is None and residual is None:
                raise ValueError(
                    "Ambiguous reaction: one of product, residual, or "
                    + "process must be provided."
                )
            # no process string, just set up reaction based on whichever of
            # product or residual are provided
            self.product = product
            self.residual = residual

            if self.product is None:
                self.product = self.compound_system - self.residual
            if self.residual is None:
                self.residual = self.compound_system - self.product

            self.Q = (
                self.projectile.m0 + self.target.m0 - self.residual.m0 - self.product.m0
            )
            if all_nuclei(self.target, self.projectile, self.residual, self.product):
                if (
                    self.target.A + self.projectile.A
                    != self.residual.A + self.product.A
                ) or (
                    self.target.Z + self.projectile.Z
                    != self.residual.Z + self.product.Z
                ):
                    raise ValueError("Isospin not conserved in this reaction.")

            self.reaction_string = (
                f"{self.target}({self.projectile},{self.product}){self.residual}"
            )
            self.reaction_latex = (
                f"{self.target.latex()}({self.projectile.latex()},"
                + f"{self.product.latex()}){self.residual.latex()}"
            )

        # get separation energy threshold and Fermi energy of projectile in target
        self.Ef = 0
        self.threshold = 0
        if all_nuclei(self.projectile, self.target):
            self.threshold = cluster_separation_energy(self.projectile, self.target, **mass_kwargs)


    def is_match(self, subentry, vocal=False):
        """Checks if the reaction matches a given subentry.

        Args:
            subentry: The subentry to match against.
            vocal (bool, optional): If True, provides verbose output. Defaults to False.

        Returns:
            bool: True if the reaction matches the subentry, False otherwise.
        """
        target = (subentry.reaction[0].targ.getA(), subentry.reaction[0].targ.getZ())
        projectile = (
            subentry.reaction[0].proj.getA(),
            subentry.reaction[0].proj.getZ(),
        )

        if target != self.target or projectile != self.projectile:
            return False

        product = subentry.reaction[0].products[0]
        if isinstance(product, str):
            if product != self.process:
                return False
        else:
            product = (product.getA(), product.getZ())
            if product != self.product:
                return False

        if subentry.reaction[0].residual is None:
            return self.residual is None
        else:
            residual = (
                subentry.reaction[0].residual.getA(),
                subentry.reaction[0].residual.getZ(),
            )
            return residual == self.residual

    def __eq__(self, other):
        """Checks equality with another Reaction instance.

        Args:
            other (Reaction): The other Reaction instance to compare.

        Returns:
            bool: True if equal, False otherwise.
        """
        if not isinstance(other, Reaction):
            return False
        return (
            self.target,
            self.projectile,
            self.product,
            self.residual,
            self.process,
        ) == (
            other.target,
            other.projectile,
            other.product,
            other.residual,
            other.process,
        )

    def __hash__(self):
        """Returns the hash of the Reaction instance.

        Returns:
            int: The hash value.
        """
        return hash((self.target, self.projectile, self.product, self.residual))


class ElasticReaction(Reaction):
    """
    Represents an elastic reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, target, projectile, **kwargs):
        super().__init__(target, projectile, None, None, "el", **kwargs)


class InelasticReaction(Reaction):
    """
    Represents an inelastic reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, target, projectile, **kwargs):
        super().__init__(target, projectile, None, None, "inl", target, **kwargs)


class TotalReaction(Reaction):
    """
    Represents a total reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, target, projectile, **kwargs):
        super().__init__(target, projectile, None, None, "tot", None, **kwargs)


class AbsorptionReaction(Reaction):
    """
    Represents an absorption reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, target, projectile, **kwargs):
        super().__init__(target, projectile, None, None, "abs", None, **kwargs)


class InclusiveReaction(Reaction):
    """
    Represents an inclusive reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        residual: The residual nucleus.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, target, projectile, residual, **kwargs):
        super().__init__(target, projectile, None, None, "x", residual, **kwargs)


class GammaCaptureReaction(Reaction):
    """
    Represents a gamma capture reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, target, projectile, **kwargs):
        residual = target + projectile
        product = Gamma()
        super().__init__(target, projectile, residual, product, **kwargs)


def get_latex(A, Z, Ex=None):
    """
    Returns the LaTeX representation of a nucleus.

    Params:
        A: Mass number.
        Z: Atomic number.
        Ex: Excitation energy (optional).
    Returns:
        LaTeX string.
    """
    if (A, Z) == (1, 0):
        return "n"
    elif (A, Z) == (1, 1):
        return "p"
    elif (A, Z) == (2, 1):
        return "d"
    elif (A, Z) == (3, 1):
        return "t"
    elif (A, Z) == (4, 2):
        return r"$\alpha$"
    else:
        if Ex is None:
            return f"$^{{{A}}}${str(periodictable.elements[Z])}"
        else:
            ex = f"({float(Ex):1.3f})"
            return f"$^{{{A}}}${str(periodictable.elements[Z])}{ex}"


def get_symbol(A, Z, Ex=None):
    """
    Returns the symbol representation of a nucleus.

    Params:
        A: Mass number.
        Z: Atomic number.
        Ex: Excitation energy (optional).
    Returns:
        Symbol string.
    """
    if (A, Z) == (1, 0):
        return "n"
    elif (A, Z) == (1, 1):
        return "p"
    elif (A, Z) == (2, 1):
        return "d"
    elif (A, Z) == (3, 1):
        return "t"
    elif (A, Z) == (4, 2):
        return r"alpha"
    else:
        if Ex is None:
            return f"{A}-{str(periodictable.elements[Z])}"
        else:
            ex = f"({float(Ex):1.3f})"
            return f"{A}-{str(periodictable.elements[Z])}{ex}"


def all_nuclei(a, b, c, d):
    for n in [a, b, c, d]:
        if not (n is None or isinstance(n, Nucleus)):
            return False
    return True


def cluster_separation_energy(target: Nucleus, projectile: Nucleus, **mass_kwargs):
    mf = mass.mass(
        target.A - projectile.A,
        target.Z - projectile.Z,
        **mass_kwargs,
    )[0]
    return mf + projectile.m0 - target.m0

