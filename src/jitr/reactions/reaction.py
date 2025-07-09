import numpy as np

from ..utils import mass, constants
from ..utils.kinematics import ChannelKinematics, semi_relativistic_kinematics
import periodictable


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
    def parse(cls, p, **kwargs):
        if p is None:
            return None
        elif isinstance(p, tuple):
            return Nucleus(*p, **kwargs)
        elif isinstance(p, Nucleus):
            return p
        elif isinstance(p, Gamma):
            return p
        elif isinstance(p, str):
            return NotImplemented
            # TODO parse from str
        else:
            return ValueError(f"Can't parse a particle from a {type(p)}")


# TODO add GS spin and excited states from ENSDF
class Nucleus(Particle):
    """
    Represents a Nucleus with atomic mass number A and atomic number Z.
    Nucleons can also be represented as a Nucleus.

    Attributes:
        A (int): Atomic mass number.
        Z (int): Atomic number.
        Efn (float): Neutron Fermi energy.
        Efp (float): Proton Fermi energy.
    """

    def __init__(self, A: int, Z: int, mass_kwargs={}):
        """
        Initializes a Nucleus instance.

        Params:
            A (int): Atomic mass number (must be greater than 0).
            Z (int): Atomic number (must be greater than or equal to 0).
            mass_kwargs: Additional keyword arguments for mass calculations.
        """

        self.A = A
        self.Z = Z
        self.mass_kwargs = mass_kwargs

        if A > 1:
            m0 = mass.mass(A, Z, **mass_kwargs)[0]
        elif A == 1 and Z == 1:
            m0 = constants.MASS_P
        elif A == 1 and Z == 0:
            m0 = constants.MASS_N
        else:
            raise ValueError(f"Cannot construct a nucleus with A={A} and Z={Z}")

        super().__init__(m0, Z)

        self.Efn = mass.neutron_fermi_energy(self.A, self.Z, **mass_kwargs)[0]
        self.Efp = mass.proton_fermi_energy(self.A, self.Z, **mass_kwargs)[0]

    def __add__(self, other):
        """
        Adds two particles together.

        Params:
            other (Nucleus): Another particle to add.

        Returns:
            Nucleus: New particle resulting from the addition.
        """
        if isinstance(other, Nucleus):
            return Nucleus(
                self.A + other.A, self.Z + other.Z, mass_kwargs=self.mass_kwargs
            )
        elif isinstance(other, tuple):
            return Nucleus(
                self.A + other[0], self.Z + other[1], mass_kwargs=self.mass_kwargs
            )
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
            return Nucleus(
                self.A - other.A, self.Z - other.Z, mass_kwargs=self.mass_kwargs
            )
        elif isinstance(other, tuple):
            return Nucleus(
                self.A - other[0], self.Z - other[1], mass_kwargs=self.mass_kwargs
            )
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
        return r"\gamma"

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
        return r"e^{-}"

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
        return r"e^{+}"

    def __repr__(self):
        return "e+:"


# TODO support multiple processes/products, e.g. (n,2NF) or (N,3N)
class Reaction:
    """Represents a 2-body nuclear reaction of the form A + a -> b + B.

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
        mass_kwargs={},
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
        self.target = Particle.parse(target, mass_kwargs=mass_kwargs)
        self.projectile = Particle.parse(projectile, mass_kwargs=mass_kwargs)
        self.compound_system = self.target + self.projectile
        self.process = process

        if product is not None:
            product = Particle.parse(product, mass_kwargs=mass_kwargs)
        if residual is not None:
            residual = Particle.parse(residual, mass_kwargs=mass_kwargs)

        # parse the process string and ensuure the reactants are consistent
        residual_in_string = True
        if process is not None:
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
                self.residual = Particle.parse(residual, mass_kwargs=mass_kwargs)
                self.Q = None

            # form of reaction strings that includes process string rather than product
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

            # most general form for Q
            self.Q = (
                self.projectile.m0 + self.target.m0 - self.residual.m0 - self.product.m0
            )
            if these_things_are_all_nuclei(
                self.target, self.projectile, self.residual, self.product
            ):
                # make sure total isospin is conserved
                if (
                    self.target.A + self.projectile.A
                    != self.residual.A + self.product.A
                ) or (
                    self.target.Z + self.projectile.Z
                    != self.residual.Z + self.product.Z
                ):
                    raise ValueError("Isospin not conserved in this reaction.")

            # form of reaction strings that explicitly identify all reactants;
            # no process string
            self.reaction_string = (
                f"{self.target}({self.projectile},{self.product}){self.residual}"
            )
            self.reaction_latex = (
                f"{self.target.latex()}({self.projectile.latex()},"
                + f"{self.product.latex()}){self.residual.latex()}"
            )

        # get separation energy threshold and Fermi energy of projectile in target
        self.Ef = None
        self.threshold = None
        self.compound_system_threshold = None
        if these_things_are_all_nuclei(self.projectile, self.target):
            self.threshold = cluster_separation_energy(
                self.target, self.projectile, **mass_kwargs
            )
            self.compound_system_threshold = cluster_separation_energy(
                self.compound_system, self.projectile, **mass_kwargs
            )
            self.Ef = -0.5 * (self.threshold + self.compound_system_threshold)

    def kinematics(self, Elab: float) -> ChannelKinematics:
        """
        Entrance channel kinematics given projectile incident on target with
        lab energy Elab in MeV.

        Params:
            Elab (float): The laboratory energy of the projectile.

        Returns:
            ChannelKinematics: the kinematics
        """
        return semi_relativistic_kinematics(
            self.target.m0,
            self.projectile.m0,
            Elab,
            Zz=self.target.Z * self.projectile.Z,
        )

    def kinematics_cm(self, Ecm: float) -> ChannelKinematics:
        """
        Entrance channel kinematics given a kinetic energy of Ecm in the
        projectile-target center-of-mass frame.

        Params:
            Ecm (float): The kinetic energy in the center-of-mass frame.

        Returns:
            ChannelKinematics: the kinematics
        """
        Elab = Ecm * (self.target.m0 + self.projectile.m0) / self.target.m0
        result = semi_relativistic_kinematics(
            self.target.m0,
            self.projectile.m0,
            Elab,
            Zz=self.target.Z * self.projectile.Z,
        )
        assert np.isclose(Ecm, result.Ecm)

        return result

    def kinematics_exit(
        self,
        entrance: ChannelKinematics,
        residual_excitation_energy: float = 0,
        product_excitation_energy: float = 0,
    ) -> ChannelKinematics:
        """
        Exit channel kinematics given entrance channel kinematics and
            excitation energies.

        Params:
            entrance (ChannelKinematics): The entrance channel kinematics.
            residual_excitation_energy (float): The excitation energy
                of the residual nucleus.
            product_excitation_energy (float): The excitation energy of the
                product nucleus.

        Returns:
            ChannelKinematics: the kinematics in the exit channel
        """
        Ecm = (
            entrance.Ecm
            + self.Q
            - residual_excitation_energy
            - product_excitation_energy
        )
        Elab = (self.residual.m0 + self.product.m0) / self.residual.m0 * Ecm
        return semi_relativistic_kinematics(
            self.residual.m0 + residual_excitation_energy,
            self.product.m0,
            Elab,
            Zz=self.residual.Z * self.product.Z,
        )

    def __repr__(self):
        """
        Returns the symbolic representation of the Reaction.

        Returns:
            str: Symbolic representation.
        """
        return self.reaction_string

    def __str__(self):
        return self.__repr__()

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


class ElasticReaction(Reaction):
    """
    Represents an elastic reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, target, projectile, **kwargs):
        super().__init__(
            target, projectile, product=None, residual=None, process="el", **kwargs
        )


class InelasticReaction(Reaction):
    """
    Represents an inelastic reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        kwargs: Additional keyword arguments for Reaction.
    """

    def __init__(self, target, projectile, **kwargs):
        super().__init__(
            target, projectile, product=None, residual=None, process="inl", **kwargs
        )


class TotalReaction(Reaction):
    """
    Represents a total reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        kwargs: Additional keyword arguments for Reaction.
    """

    def __init__(self, target, projectile, **kwargs):
        super().__init__(
            target, projectile, product=None, residual=None, process="tot", **kwargs
        )


class AbsorptionReaction(Reaction):
    """
    Represents an absorption reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        kwargs: Additional keyword arguments for Reaction.
    """

    def __init__(self, target, projectile, **kwargs):
        super().__init__(
            target, projectile, product=None, residual=None, process="abs", **kwargs
        )


class InclusiveReaction(Reaction):
    """
    Represents an inclusive reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        residual: The residual nucleus.
        kwargs: Additional keyword arguments for Reaction.
    """

    def __init__(self, target, projectile, residual, **kwargs):
        super().__init__(
            target, projectile, product=None, residual=residual, process="x", **kwargs
        )


class GammaCaptureReaction(Reaction):
    """
    Represents a gamma capture reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        kwargs: Additional keyword arguments for Reaction.
    """

    def __init__(self, target, projectile, **kwargs):
        residual = target + projectile
        super().__init__(
            target, projectile, residual=residual, product=Gamma(), **kwargs
        )


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
        return r"\alpha"
    else:
        if Ex is None:
            return f"^{{{A}}} \\rm{{{periodictable.elements[Z]}}}"
        else:
            ex = f"({float(Ex):1.3f})"
            return f"^{{{A}}} \\rm{{{periodictable.elements[Z]}}}({ex})"


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


def these_things_are_all_nuclei(*things_that_might_be_nuclei):
    for thing in things_that_might_be_nuclei:
        if not (thing is None or isinstance(thing, Nucleus)):
            return False
    return True


def cluster_separation_energy(
    target: Nucleus,
    projectile: Nucleus,
    **mass_kwargs,
):
    mf = mass.mass(
        target.A - projectile.A,
        target.Z - projectile.Z,
        **mass_kwargs,
    )[0]
    return mf + projectile.m0 - target.m0
