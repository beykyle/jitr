"""Reaction-system models and helpers for common nuclear processes."""

from __future__ import annotations

from collections.abc import Iterator

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

    :ivar m0: The rest mass of the particle in MeV/c^2.
    :vartype m0: float
    :ivar q: The electric charge of the particle.
    :vartype q: float
    :ivar Z: Integer charge number (alias for ``int(q)``).
    :vartype Z: int
    :ivar A: Mass number (0 for non-nuclear particles).
    :vartype A: int
    """

    m0: float
    q: float
    Z: int
    A: int

    def __init__(self, m0: float, q: float) -> None:
        """
        Initializes a Particle instance.

        :param m0: The rest mass of the particle in MeV/c^2.
        :type m0: float
        :param q: The electric charge of the particle.
        :type q: float
        """
        self.m0 = m0
        self.q = q
        self.Z = int(q)
        self.A = 0  # non-nuclear particles have A = 0

    def latex(self) -> str:
        """
        Returns the LaTeX representation of the particle.

        :returns: LaTeX string.
        :rtype: str
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """
        Returns the string representation of the particle.

        :returns: String representation.
        :rtype: str
        """
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """
        Checks equality with another particle.

        :param other: Another particle to compare.
        :returns: True if equal, False otherwise.
        :rtype: bool
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Returns the symbolic representation of the particle.

        :returns: Symbolic representation.
        :rtype: str
        """
        raise NotImplementedError

    def __add__(self, other: Particle | tuple[int, int]) -> Particle:
        """Add two particles; implemented in concrete subclasses."""
        raise NotImplementedError

    def __sub__(self, other: Particle | tuple[int, int]) -> Particle:
        """Subtract a particle; implemented in concrete subclasses."""
        raise NotImplementedError

    @classmethod
    def parse(
        cls,
        p: object,
        mass_kwargs: dict[str, str] | None = None,
    ) -> Particle:
        """Parse a particle-like object into a concrete particle instance.

        :param p: A ``Particle`` instance, a ``(A, Z)`` tuple, or a string.
        :param mass_kwargs: Keyword arguments forwarded to the mass database.
        :raises NotImplementedError: When ``p`` is a string (not yet supported).
        :raises ValueError: When ``p`` cannot be interpreted as a particle.
        :returns: A concrete ``Particle`` instance.
        """
        if isinstance(p, tuple):
            A, Z = int(p[0]), int(p[1])
            return Nucleus(A, Z, mass_kwargs=mass_kwargs)
        elif isinstance(p, Particle):
            return p
        elif isinstance(p, str):
            raise NotImplementedError(
                "Parsing a particle from a string is not yet supported."
            )
        else:
            raise ValueError(f"Can't parse a particle from a {type(p)}")


# TODO add GS spin and excited states from ENSDF
class Nucleus(Particle):
    """
    Represents a Nucleus with atomic mass number A and atomic number Z.
    Nucleons can also be represented as a Nucleus.

    :ivar A: Atomic mass number.
    :vartype A: int
    :ivar Z: Atomic number.
    :vartype Z: int
    :ivar Efn: Neutron Fermi energy in MeV.
    :vartype Efn: float
    :ivar Efp: Proton Fermi energy in MeV.
    :vartype Efp: float
    """

    A: int
    Z: int
    Efn: float
    Efp: float
    mass_kwargs: dict[str, str]

    def __init__(
        self,
        A: int,
        Z: int,
        mass_kwargs: dict[str, str] | None = None,
    ) -> None:
        """
        Initializes a Nucleus instance.

        :param A: Atomic mass number (must be greater than 0).
        :type A: int
        :param Z: Atomic number (must be greater than or equal to 0).
        :type Z: int
        :param mass_kwargs: Keyword arguments forwarded to the mass database
            (e.g. ``{"model": "ame2020"}``).
        """
        if mass_kwargs is None:
            mass_kwargs = {}

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
        # Override the generic A=0 set by Particle.__init__; Z is already correct.
        self.A = A

        self.Efn = mass.neutron_fermi_energy(self.A, self.Z, **mass_kwargs)[0]
        self.Efp = mass.proton_fermi_energy(self.A, self.Z, **mass_kwargs)[0]

    def __add__(self, other: Particle | tuple[int, int]) -> Nucleus:
        """
        Adds two particles together.

        :param other: Another particle to add.
        :returns: New Nucleus resulting from the addition.
        :rtype: Nucleus
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

    def __sub__(self, other: Particle | tuple[int, int]) -> Nucleus:
        """
        Subtracts a particle from this nucleus.

        :param other: Another particle to subtract.
        :returns: New Nucleus resulting from the subtraction.
        :rtype: Nucleus
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
            raise ValueError(f"Cannot subtract {type(other)} from a Nucleus")

    def latex(self) -> str:
        """
        Returns the LaTeX representation of the nucleus.

        :returns: LaTeX string.
        :rtype: str
        """
        return get_latex(self.A, self.Z)

    def __eq__(self, other: object) -> bool:
        """
        Checks equality with another particle.

        :param other: Another particle to compare.
        :returns: True if equal, False otherwise.
        :rtype: bool
        """
        if isinstance(other, Nucleus):
            return self.A == other.A and self.Z == other.Z
        elif isinstance(other, tuple):
            return self.A == other[0] and self.Z == other[1]
        else:
            return False

    def __repr__(self) -> str:
        """
        Returns the symbolic representation of the nucleus.

        :returns: Symbolic representation.
        :rtype: str
        """
        return get_symbol(self.A, self.Z)

    def __iter__(self) -> Iterator[int]:
        """
        Allows unpacking of a Nucleus instance into (A, Z).

        :returns: An iterator over the atomic mass number and atomic number.
        :rtype: Iterator[int]
        """
        return iter((self.A, self.Z))


class Gamma(Particle):
    """
    Represents a gamma-ray

    Inherits from Particle with m0=0 and q=0
    """

    def __init__(self) -> None:
        """
        Initializes a Gamma object
        """
        super().__init__(0, 0)

    def latex(self) -> str:
        return r"\gamma"

    def __repr__(self) -> str:
        return "gamma"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Gamma)


class Electron(Particle):
    """
    Represents an electron

    Inherits from Particle
    """

    def __init__(self) -> None:
        """
        Initializes an Electron
        """
        super().__init__(constants.MASS_E, -1)

    def latex(self) -> str:
        return r"e^{-}"

    def __repr__(self) -> str:
        return "e-"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Electron)


class Positron(Particle):
    """
    Represents a positron

    Inherits from Particle
    """

    def __init__(self) -> None:
        """
        Initializes a positron
        """
        super().__init__(constants.MASS_E, +1)

    def latex(self) -> str:
        return r"e^{+}"

    def __repr__(self) -> str:
        return "e+"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Positron)


# TODO support multiple processes/products, e.g. (n,2NF) or (N,3N)
class Reaction:
    """Represents a 2-body nuclear reaction of the form A + a -> b + B.

    :ivar target: The target particle.
    :vartype target: Particle
    :ivar projectile: The projectile particle.
    :vartype projectile: Particle
    :ivar product: The light product, or None for absorption/total processes.
    :vartype product: Particle or None
    :ivar residual: The residual particle, or None for 'abs' or 'tot' processes.
    :vartype residual: Particle or None
    :ivar compound_system: The compound system formed by target and projectile.
    :vartype compound_system: Particle
    :ivar process: Short process label (e.g. ``'el'``, ``'abs'``), or ``None``.
    :vartype process: str or None
    :ivar Q: The Q-value of the reaction in MeV. ``None`` when unspecified.
    :vartype Q: float or None
    :ivar reaction_string: The string representation of the reaction.
    :vartype reaction_string: str
    :ivar reaction_latex: The LaTeX representation of the reaction.
    :vartype reaction_latex: str
    :ivar Ef: Effective Fermi energy in MeV, or ``None`` for non-nuclear reactions.
    :vartype Ef: float or None
    :ivar threshold: Cluster separation energy threshold in MeV, or ``None``.
    :vartype threshold: float or None
    :ivar compound_system_threshold: Compound-system separation energy in MeV, or ``None``.
    :vartype compound_system_threshold: float or None
    """

    target: Particle
    projectile: Particle
    compound_system: Particle
    product: Particle | None
    residual: Particle | None
    process: str | None
    Q: float | None
    reaction_string: str
    reaction_latex: str
    Ef: float | None
    threshold: float | None
    compound_system_threshold: float | None

    def __init__(
        self,
        target: Particle | tuple[int, int],
        projectile: Particle | tuple[int, int],
        product: Particle | tuple[int, int] | None = None,
        residual: Particle | tuple[int, int] | None = None,
        process: str | None = None,
        mass_kwargs: dict[str, str] | None = None,
    ) -> None:
        """Initializes a Reaction instance.

        :param target: The target particle.
        :type target: Particle
        :param projectile: The projectile particle.
        :type projectile: Particle
        :param product: The light product particle, or ``None``.
        :type product: Particle or None
        :param process: Short process label. Currently supported are ``'tot'``,
            ``'el'``, ``'inl'``, ``'abs'``, ``'x'``, ``'sct'``, ``'non'``, and
            ``'f'``. Process types are defined in the EXFOR manual:
            https://www-nds.iaea.org/nrdc/nrdc_doc/bnl-ncs-063380-200105.pdf
        :type process: str or None
        :param residual: The residual particle, or None for 'abs' or 'tot'
                         processes.
        :type residual: Particle or None
        :param mass_kwargs: Keyword arguments forwarded to the mass database
            (e.g. ``{"model": "ame2020"}``).
        :raises ValueError: If isospin is not conserved or if invalid product/
                            residual types are provided.
        """
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

        # parse the process string and ensure the reactants are consistent
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
                self.Q = 0.0
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
                # residual_in_string=True only for 'x' process, where residual is set
                assert self.residual is not None
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
                assert self.residual is not None
                self.product = self.compound_system - self.residual
            if self.residual is None:
                assert self.product is not None
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
        :rtype: ChannelKinematics
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
        :rtype: ChannelKinematics
        """
        if self.Q is None:
            raise ValueError(
                "kinematics_exit() requires a defined Q-value. "
                f"This reaction ({self.reaction_string}) has Q=None."
            )
        if self.residual is None or self.product is None:
            raise ValueError(
                "kinematics_exit() requires both product and residual to be defined. "
                f"This reaction ({self.reaction_string}) does not have a defined "
                "product and/or residual."
            )
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

    def __repr__(self) -> str:
        """
        Returns the symbolic representation of the Reaction.

        :returns: Symbolic representation.
        :rtype: str
        """
        return self.reaction_string

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Checks equality with another Reaction instance.

        :param other: The other Reaction instance to compare.
        :returns: True if equal, False otherwise.
        :rtype: bool
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

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param mass_kwargs: Keyword arguments forwarded to the mass database.
    """

    def __init__(
        self,
        target: Particle | tuple[int, int],
        projectile: Particle | tuple[int, int],
        mass_kwargs: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            target,
            projectile,
            product=None,
            residual=None,
            process="el",
            mass_kwargs=mass_kwargs,
        )


class InelasticReaction(Reaction):
    """
    Represents an inelastic reaction.

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param mass_kwargs: Keyword arguments forwarded to the mass database.
    """

    def __init__(
        self,
        target: Particle | tuple[int, int],
        projectile: Particle | tuple[int, int],
        mass_kwargs: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            target,
            projectile,
            product=None,
            residual=None,
            process="inl",
            mass_kwargs=mass_kwargs,
        )


class TotalReaction(Reaction):
    """
    Represents a total reaction.

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param mass_kwargs: Keyword arguments forwarded to the mass database.
    """

    def __init__(
        self,
        target: Particle | tuple[int, int],
        projectile: Particle | tuple[int, int],
        mass_kwargs: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            target,
            projectile,
            product=None,
            residual=None,
            process="tot",
            mass_kwargs=mass_kwargs,
        )


class AbsorptionReaction(Reaction):
    """
    Represents an absorption reaction.

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param mass_kwargs: Keyword arguments forwarded to the mass database.
    """

    def __init__(
        self,
        target: Particle | tuple[int, int],
        projectile: Particle | tuple[int, int],
        mass_kwargs: dict[str, str] | None = None,
    ) -> None:
        _target = Particle.parse(target, mass_kwargs=mass_kwargs)
        _projectile = Particle.parse(projectile, mass_kwargs=mass_kwargs)
        residual = _target + _projectile
        super().__init__(
            target,
            projectile,
            residual=residual,
            product=None,
            process="abs",
            mass_kwargs=mass_kwargs,
        )


class InclusiveReaction(Reaction):
    """
    Represents an inclusive reaction.

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param residual: The residual nucleus.
    :param mass_kwargs: Keyword arguments forwarded to the mass database.
    """

    def __init__(
        self,
        target: Particle | tuple[int, int],
        projectile: Particle | tuple[int, int],
        residual: Particle | tuple[int, int],
        mass_kwargs: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            target,
            projectile,
            product=None,
            residual=residual,
            process="x",
            mass_kwargs=mass_kwargs,
        )


class GammaCaptureReaction(Reaction):
    """
    Represents a gamma capture reaction.

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param mass_kwargs: Keyword arguments forwarded to the mass database.
    """

    def __init__(
        self,
        target: Particle | tuple[int, int],
        projectile: Particle | tuple[int, int],
        mass_kwargs: dict[str, str] | None = None,
    ) -> None:
        _target = Particle.parse(target, mass_kwargs=mass_kwargs)
        _projectile = Particle.parse(projectile, mass_kwargs=mass_kwargs)
        residual = _target + _projectile
        super().__init__(
            target,
            projectile,
            residual=residual,
            product=Gamma(),
            mass_kwargs=mass_kwargs,
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
    target: Particle,
    projectile: Particle,
    **mass_kwargs: str,
) -> float:
    """Return the separation energy for removing ``projectile`` from ``target``."""
    mf = mass.mass(
        target.A - projectile.A,
        target.Z - projectile.Z,
        **mass_kwargs,
    )[0]
    return mf + projectile.m0 - target.m0
