import operator
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import pyteomics.cmass as pmass
except ImportError:
    import pyteomics.mass as pmass

from spectrum_utils import proforma


# Amino acid and special amino acid masses.
AA_MASS = {
    **pmass.std_aa_mass,
    # Aspartic acid / asparagine (ambiguous mass).
    # "B": 0,
    # Glutamic acid / glutamine (ambiguous mass).
    # "Z": 0,
    # Leucine / isoleucine.
    "J": 113.084_064,
    # Selenocysteine (in Pyteomics).
    # "U": 150.95363,
    # Pyrrolysine (in Pyteomics).
    # "O": 237.14772,
    # Any amino acid, gaps (zero mass).
    "X": 0,
}

# Offset for isotopic peaks.
C13_MASS_DIFF = 1.003_354

# Common neutral losses.
NEUTRAL_LOSS = {
    # No neutral loss.
    None: 0,
    # Hydrogen.
    "H": -1.007_825,
    # Ammonia.
    "NH3": -17.026_549,
    # Water.
    "H2O": -18.010_565,
    # Carbon monoxide.
    "CO": -27.994_915,
    # Carbon dioxide.
    "CO2": -43.989_829,
    # Formamide.
    "HCONH2": -45.021_464,
    # Formic acid.
    "HCOOH": -46.005_479,
    # Methanesulfenic acid.
    "CH4OS": -63.998_301,
    # Sulfur trioxide.
    "SO3": -79.956_818,
    # Metaphosphoric acid.
    "HPO3": -79.966_331,
    # Mercaptoacetamide.
    "C2H5NOS": -91.009_195,
    # Mercaptoacetic acid.
    "C2H4O2S": -91.993_211,
    # Phosphoric acid.
    "H3PO4": -97.976_896,
}

SUPPORTED_IONS = "?abcxyzIm_prf"


class FragmentAnnotation:
    def __init__(
        self,
        ion_type: str,
        neutral_loss: Optional[str] = None,
        isotope: int = 0,
        charge: Optional[int] = None,
        adduct: Optional[str] = None,
        analyte_number: Optional[int] = None,
        mz_delta: Optional[Tuple[float, str]] = None,
    ) -> None:
        """
        Individual fragment ion annotation.

        This fragment annotation format is derived from the PSI peak
        interpretation specification:
        https://docs.google.com/document/d/1yEUNG4Ump6vnbMDs4iV4s3XISflmOkRAyqUuutcCG2w/edit?usp=sharing

        Fragment notations have the following format:

        (analyte_number)[ion_type](neutral_loss)(isotope)(charge)(adduct)(mz_delta)

        Examples:

        - "y4-H2O+2i^2[M+H+Na]" : Fragment annotation for a y4 ion, with a
          water neutral loss, the second isotopic peak, charge 2, adduct
          [M+H+Na].

        Parameters
        ----------
        ion_type : str
            Specifies the basic type of ion being described.
            Possible prefixes are:

            - "?": unknown ion
            - "a", "b", "c", "x", "y", "z": corresponding peptide fragments
            - "I": immonium ion
            - "m": internal fragment ion
            - "_": named compound
            - "p": precursor ion
            - "r": reporter ion (isobaric label)
            - "f": chemical formula
        neutral_loss : Optional[str]
            A string of neutral loss(es), described by their molecular formula.
            The default is no neutral loss. Note that the neutral loss string
            must include the sign (typically "-" for a neutral loss).
        isotope : int
            The isotope number above or below the monoisotope. The default is
            the monoisotopic peak (0).
        charge : Optional[int]
            The charge of the fragment. The default is an unknown charge (only
            valid for unknown ions).
        adduct : Optional[str]
            The adduct that ionized the fragment. The default is a hydrogen
            adduct matching the charge ([M+xH]).
        mz_delta : Optional[Tuple[float, str]]
            The m/z delta representing the observed m/z minus the theoretical
            m/z and its unit ("Da" or "ppm").
        """
        if ion_type[0] in "GLXS":
            raise NotImplementedError(
                "Advanced ion types are not yet supported"
            )
        elif ion_type[0] not in SUPPORTED_IONS:
            raise ValueError("Unknown ion type")
        if ion_type == "?" and (
            neutral_loss is not None
            or isotope != 0
            or charge is not None
            or adduct is not None
            or analyte_number is not None
            or mz_delta is not None
        ):
            raise ValueError(
                "Unknown ions should not contain additional information"
            )
        self.ion_type = ion_type
        self.neutral_loss = neutral_loss
        self.isotope = isotope
        self.charge = charge
        self.adduct = f"[M+{self.charge}H]" if adduct is None else adduct
        self.analyte_number = analyte_number
        self.mz_delta = mz_delta

    @property
    def mz_delta(self) -> Optional[Tuple[float, str]]:
        return self._mz_delta

    @mz_delta.setter
    def mz_delta(self, mz_delta: Optional[Tuple[float, str]]):
        if mz_delta is not None and mz_delta[1] not in ("Da", "ppm"):
            raise ValueError(
                "The m/z delta must be specified in Dalton or ppm units"
            )
        self._mz_delta = mz_delta

    @property
    def charge(self) -> Optional[int]:
        return self._charge

    @charge.setter
    def charge(self, charge: Optional[int]):
        if self.ion_type == "?" and charge is not None:
            raise ValueError("Invalid charge for unknown ions")
        elif self.ion_type != "?" and (charge is None or charge <= 0):
            raise ValueError(
                "The charge must be specified and strictly positive for known "
                "ion types"
            )
        self._charge = charge

    def __repr__(self):
        return str(self)

    def __str__(self) -> str:
        if self.ion_type == "?":
            return "?"
        else:
            annot_str = []
            if self.analyte_number is not None:
                annot_str.append(f"{self.analyte_number}@")
            annot_str.append(self.ion_type)
            if self.neutral_loss is not None:
                annot_str.append(self.neutral_loss)
            if abs(self.isotope) == 1:
                annot_str.append("+i" if self.isotope > 0 else "-i")
            elif self.isotope != 0:
                annot_str.append(f"{self.isotope:+}i")
            if self.charge is not None and self.charge > 1:
                annot_str.append(f"^{self.charge}")
            if re.match(r"\[M\+\d+H\]", self.adduct) is None:
                annot_str.append(self.adduct)
            if self.mz_delta is not None:
                mz_delta, unit = self.mz_delta
                decimals = 1 if unit == "ppm" else 5
                annot_str.append(
                    f"/{round(mz_delta, decimals)}"
                    f"{'ppm' if unit == 'ppm' else ''}"
                )
            return "".join(annot_str)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FragmentAnnotation):
            return False
        return (
            self.ion_type == other.ion_type
            and self.neutral_loss == other.neutral_loss
            and self.isotope == other.isotope
            and self.charge == other.charge
            and self.adduct == other.adduct
            and self.analyte_number == other.analyte_number
            and self.mz_delta == other.mz_delta
        )


class PeakInterpretation:
    _unknown = FragmentAnnotation("?")

    def __init__(self):
        """
        Fragment annotation(s) to interpret a specific peak.
        """
        self.fragment_annotations = []

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        # If no fragment annotations have been specified, interpret as an
        # unknown ion.
        if len(self.fragment_annotations) > 0:
            return ",".join([str(a) for a in self.fragment_annotations])
        else:
            return str(self._unknown)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, PeakInterpretation) and str(self) == str(
            other
        )

    def __getitem__(self, key) -> FragmentAnnotation:
        if len(self.fragment_annotations) > 0:
            return self.fragment_annotations[key]
        else:
            return self._unknown


def get_theoretical_fragments(
    proteoform: proforma.Proteoform,
    ion_types: str = "by",
    *,
    max_isotope: int = 0,
    max_charge: int = 1,
    neutral_losses: Optional[Dict[Optional[str], float]] = None,
) -> List[Tuple[FragmentAnnotation, float]]:
    """
    Get fragment annotations with their theoretical masses for the given
    sequence.

    Parameters
    ----------
    proteoform : proforma.Proteoform
        The proteoform for which the fragment annotations will be
        generated.
    ion_types : str
        The ion types to generate. Can be any combination of 'a', 'b',
        'c', 'x', 'y', and 'z' for peptide fragments, 'I' for immonium
        ions, 'm' for internal fragment ions, 'p' for the precursor ion,
        and 'r' for reporter ions. The default is 'by', which means that
        b and y peptide ions will be generated.
    max_isotope : int
        The maximum isotope to consider (the default is 0 to only
        generate the monoisotopic peaks).
    max_charge : int
        All fragments up to and including the given charge will be
        generated (the default is 1 to only generate singly-charged
        fragments).
    neutral_losses : Optional[Dict[Optional[str], float]]
        A dictionary with neutral loss names and (negative) mass
        differences to be considered.

    Returns
    -------
    List[Tuple[FragmentAnnotation, float]]
        All possible fragment annotations and their theoretical m/z in
        ascending m/z order.
    """
    for ion_type in ion_types:
        if ion_type not in SUPPORTED_IONS:
            raise ValueError(
                f"{ion_type} is not a supported ion type ({SUPPORTED_IONS})"
            )
    if "B" in proteoform.sequence:
        raise ValueError(
            "Explicitly specify aspartic acid (D) or asparagine (N) instead of"
            " the ambiguous B to compute the fragment annotations"
        )
    if "Z" in proteoform.sequence:
        raise ValueError(
            "Explicitly specify glutamic acid (E) or glutamine (Q) instead of "
            "the ambiguous Z to compute the fragment annotations"
        )

    neutral_losses = {None: 0} if neutral_losses is None else neutral_losses

    base_fragments = []

    # Generate all peptide fragments ('a', 'b', 'c', 'x', 'y', 'z') and
    # calculate their theoretical masses.
    # Generate all N-terminal peptide fragments.
    for ion_type in set("abc") & set(ion_types):
        mod_i, mod_mass = 0, 0
        for fragment_i in range(1, len(proteoform.sequence)):
            fragment_sequence = proteoform.sequence[:fragment_i]
            # Ignore unlocalized modifications.
            while (
                proteoform.modifications is not None
                and mod_i < len(proteoform.modifications)
                and isinstance(proteoform.modifications[mod_i].position, str)
                and proteoform.modifications[mod_i].position != "N-term"
            ):
                mod_i += 1
            # Include prefix modifications.
            while (
                proteoform.modifications is not None
                and mod_i < len(proteoform.modifications)
                and (
                    proteoform.modifications[mod_i].position == "N-term"
                    or (
                        isinstance(
                            proteoform.modifications[mod_i].position, int
                        )
                        and proteoform.modifications[mod_i].position
                        < fragment_i
                    )
                )
            ):
                mod_mass += proteoform.modifications[mod_i].mass
                mod_i += 1
            base_fragments.append(
                (fragment_sequence, ion_type, fragment_i, mod_mass)
            )
    # Generate all C-terminal peptide fragments.
    for ion_type in set("xyz") & set(ion_types):
        if proteoform.modifications is not None:
            mod_i, mod_mass = len(proteoform.modifications) - 1, 0
        else:
            mod_i, mod_mass = None, 0
        for fragment_i in range(len(proteoform.sequence) - 1, 0, -1):
            fragment_sequence = proteoform.sequence[fragment_i:]
            # Include suffix modifications.
            while (
                proteoform.modifications is not None
                and mod_i >= 0
                and (
                    proteoform.modifications[mod_i].position == "C-term"
                    or (
                        isinstance(
                            proteoform.modifications[mod_i].position, int
                        )
                        and proteoform.modifications[mod_i].position
                        >= fragment_i
                    )
                )
            ):
                mod_mass += proteoform.modifications[mod_i].mass
                mod_i -= 1
            base_fragments.append(
                (
                    fragment_sequence,
                    ion_type,
                    len(proteoform.sequence) - fragment_i,
                    mod_mass,
                )
            )

    # Generate all internal fragment ions.
    if "m" in ion_types:
        # Skip internal fragments with start position 1, which are
        # actually b ions.
        for start_i in range(1, len(proteoform.sequence)):
            mod_i_start, mod_mass = 0, 0
            # Skip unlocalized and prefix modifications.
            while (
                proteoform.modifications is not None
                and mod_i_start < len(proteoform.modifications)
                and (
                    isinstance(
                        proteoform.modifications[mod_i_start].position, str
                    )
                    or proteoform.modifications[mod_i_start].position < start_i
                )
            ):
                mod_i_start += 1
            mod_i_stop = mod_i_start
            # Internal fragments of only one residue are encoded as
            # immonium ions.
            for stop_i in range(start_i + 2, len(proteoform.sequence)):
                fragment_sequence = proteoform.sequence[start_i:stop_i]
                # Include internal modifications.
                while (
                    proteoform.modifications is not None
                    and mod_i_stop < len(proteoform.modifications)
                    and proteoform.modifications[mod_i_stop].position < stop_i
                ):
                    mod_mass += proteoform.modifications[mod_i_stop].mass
                    mod_i_stop += 1
                # Internal fragment mass calculation is equivalent to b
                # ion mass calculation.
                base_fragments.append(
                    (
                        fragment_sequence,
                        "b",
                        f"{start_i + 1}:{stop_i + 1}",
                        mod_mass,
                    )
                )

    # Generate unfragmented precursor ion(s).
    if "p" in ion_types:
        if proteoform.modifications is not None:
            mod_mass = sum([mod.mass for mod in proteoform.modifications])
        else:
            mod_mass = 0
        base_fragments.append((proteoform.sequence, "M", "p", mod_mass))

    fragments_masses = []
    # Compute the theoretical fragment masses (using Pyteomics)
    for fragment_sequence, ion_type, fragment_i, mod_mass in base_fragments:
        for charge in range(1, max_charge + 1):
            annot_type = "?"
            if isinstance(fragment_i, str):
                if ":" in fragment_i:
                    annot_type = f"m{fragment_i}"
                elif fragment_i == "p":
                    annot_type = "p"
            else:
                annot_type = f"{ion_type}{fragment_i}"
            fragments_masses.append(
                (
                    FragmentAnnotation(ion_type=annot_type, charge=charge),
                    pmass.fast_mass(
                        sequence=fragment_sequence,
                        ion_type=ion_type,
                        charge=charge,
                        aa_mass=AA_MASS,
                    )
                    + mod_mass / charge,
                )
            )

    # Generate all immonium ions (internal single amino acid from the
    # combination of a type and y type cleavage).
    if "I" in ion_types:
        # Amino acid mass minus CO plus charge 1.
        mass_diff = pmass.calculate_mass(formula="CO") - pmass.calculate_mass(
            formula="H"
        )
        for aa, mass in AA_MASS.items():
            if aa != "X":
                fragments_masses.append(
                    (
                        FragmentAnnotation(ion_type=f"I{aa}", charge=1),
                        mass - mass_diff,
                    )
                )

    # Generate isotopic peaks for all fragments.
    isotope_fragments = []
    for isotope in range(1, max_isotope + 1):
        for fragment, mass in fragments_masses:
            isotope_fragments.append(
                (
                    FragmentAnnotation(
                        ion_type=fragment.ion_type,
                        isotope=isotope,
                        charge=fragment.charge,
                    ),
                    mass + isotope * C13_MASS_DIFF / fragment.charge,
                )
            )
    fragments_masses.extend(isotope_fragments)

    # Generate all fragments that differ by a neutral loss from the base
    # fragments.
    neutral_loss_fragments = []
    for neutral_loss, mass_diff in neutral_losses.items():
        if neutral_loss is None:
            continue
        neutral_loss = f"{'-' if mass_diff < 0 else '+'}{neutral_loss}"
        for fragment, mass in fragments_masses:
            if (fragment_mass := mass + mass_diff / fragment.charge) > 0:
                neutral_loss_fragments.append(
                    (
                        FragmentAnnotation(
                            ion_type=fragment.ion_type,
                            neutral_loss=neutral_loss,
                            isotope=fragment.isotope,
                            charge=fragment.charge,
                        ),
                        fragment_mass,
                    )
                )
    fragments_masses.extend(neutral_loss_fragments)

    # Sort the fragment annotations by their theoretical masses.
    return sorted(fragments_masses, key=operator.itemgetter(1))
