"""Reaction-system models and helpers for common nuclear processes."""

import numpy as np
import periodictable

from ..utils import constants, mass
from ..utils.kinematics import (
    ChannelKinematics,
    cm_to_lab_frame,
    lab_to_cm_frame,
    semi_relativistic_kinematics,
)


class Particle:
    """
    Represents a particle with rest mass and charge.

    :ivar m0: The rest mass of the particle.
    :vartype m0: float"""

    def __init__(self, m0, q):
        """
        Initializes a Particle instance.

        :param m0: The rest mass of the particle.
        :type m0: float
        :param q: The electric charge of the particle.
        :type q: float"""
        self.m0 = m0
        self.q = q

    def latex(self) -> str:
        """
        Returns the LaTeX representation of the particle.

        :returns: LaTeX string.
        :rtype: str"""
        pass

    def __str__(self):
        """
        Returns the string representation of the particle.

        :returns: String representation.
        :rtype: str"""
        return self.__repr__()

    def __eq__(self, other):
        """
        Checks equality with another particle.

        :param other: Another particle to compare.
        :type other: Particle
        :returns: True if equal, False otherwise.
        :rtype: bool"""
        pass

    def __repr__(self):
        """
        Returns the symbolic representation of the particle.

        :returns: Symbolic representation.
        :rtype: str"""
        pass

    @classmethod
    def parse(cls, p: object, **kwargs: object) -> object:
        """Parse a particle-like object into a concrete particle instance."""
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

    :ivar A: Atomic mass number.
    :vartype A: int
    :ivar Z: Atomic number.
    :vartype Z: int
    :ivar Efn: Neutron Fermi energy.
    :vartype Efn: float
    :ivar Efp: Proton Fermi energy.
    :vartype Efp: float"""

    def __init__(self, A: int, Z: int, mass_kwargs: dict | None = None):
        """
        Initializes a Nucleus instance.

        :param A: Atomic mass number (must be greater than 0).
        :type A: int
        :param Z: Atomic number (must be greater than or equal to 0).
        :type Z: int
        :param mass_kwargs: Additional keyword arguments for mass calculations."""

        if mass_kwargs is None:
            mass_kwargs = {}

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

        :param other: Another particle to add.
        :type other: Nucleus
        :returns: New particle resulting from the addition.
        :rtype: Nucleus"""
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

        :param other: Another particle to subtract.
        :type other: Nucleus
        :returns: New particle resulting from the subtraction.
        :rtype: Nucleus"""
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

    def latex(self) -> str:
        """
        Returns the LaTeX representation of the particle.

        :returns: LaTeX string.
        :rtype: str"""
        return get_latex(self.A, self.Z)

    def __eq__(self, other):
        """
        Checks equality with another particle.

        :param other: Another particle to compare.
        :type other: Particle
        :returns: True if equal, False otherwise.
        :rtype: bool"""
        if isinstance(other, Nucleus):
            return self.A == other.A and self.Z == other.Z
        elif isinstance(other, tuple):
            return self.A == other[0] and self.Z == other[1]
        else:
            return False

    def __repr__(self):
        """
        Returns the symbolic representation of the particle.

        :returns: Symbolic representation.
        :rtype: str"""
        return get_symbol(self.A, self.Z)

    def __iter__(self):
        """
        Allows unpacking of a Nucleus instance into (A, Z).

        :returns: An iterator over the atomic mass number and atomic number.
        :rtype: iterator"""
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

    def latex(self) -> str:
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

    def latex(self) -> str:
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

    def latex(self) -> str:
        return r"e^{+}"

    def __repr__(self):
        return "e+:"


# TODO support multiple processes/products, e.g. (n,2NF) or (N,3N)
class Reaction:
    """Represents a 2-body nuclear reaction of the form A + a -> b + B.

    :ivar target: The target particle.
    :vartype target: Particle
    :ivar projectile: The projectile particle.
    :vartype projectile: Particle
    :ivar product:
    :vartype product: Particle
    :ivar residual: The residual particle, or None for 'abs' or 'tot' processes.
    :vartype residual: Particle or None
    :ivar compound_system: The compound system formed by target and projectile.
    :vartype compound_system: Particle
    :ivar process:
    :vartype process: str
    :ivar Q: The Q-value of the reaction. If Q is None, that means it is unspecified
             for the given process.
    :vartype Q: float or None
    :ivar reaction_string: The string representation of the reaction.
    :vartype reaction_string: str
    :ivar reaction_latex: The LaTeX representation of the reaction.
    :vartype reaction_latex: str"""

    def __init__(
        self,
        target: Particle,
        projectile: Particle,
        product: Particle = None,
        residual: Particle = None,
        process: str = None,
        mass_kwargs: dict | None = None,
    ):
        """Initializes a Reaction instance.

        :param target: The target particle.
        :type target: Particle
        :param projectile: The projectile particle.
        :type projectile: Particle
        :param product:
        :type product: Particle
        :param process: The product particle or a string denoting the process.
                        Currently supported are 'tot', 'el', 'inl', 'abs', 'x',
                        'sct', 'non', and 'f'. Process types are defined in the
                        EXFOR manual:
                        https://www-nds.iaea.org/nrdc/nrdc_doc/bnl-ncs-063380-200105.pdf
        :type process: str
        :param residual: The residual particle, or None for 'abs' or 'tot'
                         processes.
        :type residual: Particle or None
        :raises ValueError: If isospin is not conserved or if invalid product/
                            residual types are provided."""
        if mass_kwargs is None:
            mass_kwargs = {}

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
                        + "\nThere must be a residual provided"
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

        :param Elab: The laboratory energy of the projectile.
        :type Elab: float
        :returns: the kinematics
        :rtype: ChannelKinematics"""
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

        :param Ecm: The kinetic energy in the center-of-mass frame.
        :type Ecm: float
        :returns: the kinematics
        :rtype: ChannelKinematics"""
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

        :param entrance: The entrance channel kinematics.
        :type entrance: ChannelKinematics
        :param residual_excitation_energy: The excitation energy of the residual
                                           nucleus.
        :type residual_excitation_energy: float
        :param product_excitation_energy: The excitation energy of the product
                                          nucleus.
        :type product_excitation_energy: float
        :returns: the kinematics in the exit channel
        :rtype: ChannelKinematics"""
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

    def to_lab_frame(self, theta_cm: np.ndarray, Ecm: float, Q: float) -> np.ndarray:
        """
        Convert angles from the center-of-mass frame to the laboratory frame
        (target rest frame).
        :param theta_cm: Angles in the center-of-mass frame in degrees.
        :type theta_cm: np.ndarray
        :param Ecm: Center-of-mass energy.
        :type Ecm: float
        :param Q: Q-value of the reaction.
        :type Q: float"""
        if self.product is None or self.residual is None:
            raise ValueError(
                "to_lab_frame() requires both product and residual to be defined. "
                f"This reaction ({self.reaction_string}) does not have a defined "
                "product and/or residual."
            )
        return cm_to_lab_frame(
            theta_cm,
            self.projectile.m0,
            self.target.m0,
            self.product.m0,
            self.residual.m0,
            Ecm,
            Q,
        )

    def to_cm_frame(self, theta_lab: np.ndarray, Elab: float, Q: float) -> np.ndarray:
        """
        Convert angles from the laboratory (target rest frame) frame to the
        center-of-mass frame.
        :param theta_lab: Angles in the laboratory frame in degrees.
        :type theta_lab: np.ndarray
        :param Elab: Laboratory energy.
        :type Elab: float
        :param Q: Q-value of the reaction.
        :type Q: float"""
        if self.product is None or self.residual is None:
            raise ValueError(
                "to_cm_frame() requires both product and residual to be defined. "
                f"This reaction ({self.reaction_string}) does not have a defined "
                "product and/or residual."
            )
        return lab_to_cm_frame(
            theta_lab,
            self.projectile.m0,
            self.target.m0,
            self.product.m0,
            self.residual.m0,
            Elab,
            Q,
        )

    def __repr__(self):
        """
        Returns the symbolic representation of the Reaction.

        :returns: Symbolic representation.
        :rtype: str"""
        return self.reaction_string

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        """Checks equality with another Reaction instance.

        :param other: The other Reaction instance to compare.
        :type other: Reaction
        :returns: True if equal, False otherwise.
        :rtype: bool"""
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

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param kwargs: Additional keyword arguments."""

    def __init__(self, target, projectile, **kwargs):
        super().__init__(
            target, projectile, product=None, residual=None, process="el", **kwargs
        )


class InelasticReaction(Reaction):
    """
    Represents an inelastic reaction.

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param kwargs: Additional keyword arguments for Reaction."""

    def __init__(self, target, projectile, **kwargs):
        super().__init__(
            target, projectile, product=None, residual=None, process="inl", **kwargs
        )


class TotalReaction(Reaction):
    """
    Represents a total reaction.

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param kwargs: Additional keyword arguments for Reaction."""

    def __init__(self, target, projectile, **kwargs):
        super().__init__(
            target, projectile, product=None, residual=None, process="tot", **kwargs
        )


class AbsorptionReaction(Reaction):
    """
    Represents an absorption reaction.

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param kwargs: Additional keyword arguments for Reaction."""

    def __init__(self, target, projectile, **kwargs):
        residual = Nucleus(
            *target, mass_kwargs=kwargs.get("mass_kwargs", None)
        ) + Nucleus(*projectile, mass_kwargs=kwargs.get("mass_kwargs", None))
        super().__init__(
            target, projectile, residual=residual, product=None, process="abs", **kwargs
        )


class InclusiveReaction(Reaction):
    """
    Represents an inclusive reaction.

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param residual: The residual nucleus.
    :param kwargs: Additional keyword arguments for Reaction."""

    def __init__(self, target, projectile, residual, **kwargs):
        super().__init__(
            target, projectile, product=None, residual=residual, process="x", **kwargs
        )


class GammaCaptureReaction(Reaction):
    """
    Represents a gamma capture reaction.

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param kwargs: Additional keyword arguments for Reaction."""

    def __init__(self, target, projectile, **kwargs):
        residual = Nucleus(
            *target, mass_kwargs=kwargs.get("mass_kwargs", None)
        ) + Nucleus(*projectile, mass_kwargs=kwargs.get("mass_kwargs", None))
        super().__init__(
            target, projectile, residual=residual, product=Gamma(), **kwargs
        )


def get_latex(A: int, Z: int, Ex: float | None = None) -> str:
    """
    Returns the LaTeX representation of a nucleus.

    :param A: Mass number.
    :param Z: Atomic number.
    :param Ex: Excitation energy (optional).
    :returns:
    :rtype: LaTeX string."""
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


def get_symbol(A: int, Z: int, Ex: float | None = None) -> str:
    """
    Returns the symbol representation of a nucleus.

    :param A: Mass number.
    :param Z: Atomic number.
    :param Ex: Excitation energy (optional).
    :returns:
    :rtype: Symbol string."""
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


def these_things_are_all_nuclei(*things_that_might_be_nuclei: object) -> bool:
    """Return ``True`` when each supplied object is a nucleus or ``None``."""
    for thing in things_that_might_be_nuclei:
        if not (thing is None or isinstance(thing, Nucleus)):
            return False
    return True


def cluster_separation_energy(
    target: Nucleus,
    projectile: Nucleus,
    **mass_kwargs: object,
) -> float:
    """Return the separation energy for removing ``projectile`` from ``target``."""
    mf = mass.mass(
        target.A - projectile.A,
        target.Z - projectile.Z,
        **mass_kwargs,
    )[0]
    return mf + projectile.m0 - target.m0
