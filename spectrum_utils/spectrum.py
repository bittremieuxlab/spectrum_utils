from __future__ import annotations

import copy
import functools
import urllib.parse
from typing import Dict, Iterable, Optional, Union

import numba as nb
import numpy as np
import pyteomics.usi

from spectrum_utils import fragment_annotation as fa, proforma, utils


class GnpsBackend(pyteomics.usi._PROXIBackend):
    _url_template = (
        "https://metabolomics-usi.gnps2.org/proxi/v{version}/spectra?usi={usi}"
    )

    def __init__(self, **kwargs):
        super(GnpsBackend, self).__init__("GNPS", self._url_template, **kwargs)


# Reload the Pyteomics PROXI aggregator to also include GNPS.
try:
    pyteomics.usi._proxies["gnps"] = GnpsBackend
    pyteomics.usi.AGGREGATOR = pyteomics.usi.PROXIAggregator()
except (TypeError, AttributeError):
    # Newer version of pyteomics - try alternative approach
    try:
        if hasattr(pyteomics.usi, "register_proxi_backend"):
            pyteomics.usi.register_proxi_backend("gnps", GnpsBackend)
        else:
            # Fallback: create aggregator with GNPS backend directly
            backends = {"gnps": GnpsBackend()}
            if hasattr(pyteomics.usi, "_proxies"):
                backends.update(pyteomics.usi._proxies)
            pyteomics.usi.AGGREGATOR = pyteomics.usi.PROXIAggregator(
                backends=backends
            )
    except Exception:
        # If all else fails, use default aggregator without GNPS
        pyteomics.usi.AGGREGATOR = pyteomics.usi.PROXIAggregator()


@nb.experimental.jitclass
class MsmsSpectrumJit:
    """
    Helper MsmsSpectrum class for JIT-compiled fast data processing.
    """

    identifier: nb.types.string
    precursor_mz: nb.float64
    precursor_charge: nb.int8
    _mz: nb.float64[:]
    _intensity: nb.float32[:]
    retention_time: nb.float32

    def __init__(
        self,
        identifier: str,
        precursor_mz: float,
        precursor_charge: int,
        mz: np.ndarray,
        intensity: np.ndarray,
        retention_time: float,
        _skip_checks: bool = False,
    ) -> None:
        self.identifier = identifier
        self.precursor_mz = precursor_mz
        self.precursor_charge = precursor_charge
        if not _skip_checks and len(mz) != len(intensity):
            raise ValueError(
                "The m/z and intensity arrays should have equal lengths"
            )
        self._mz = np.asarray(mz, np.float64).reshape(-1)
        # Make sure the peaks are sorted by m/z.
        self._intensity = np.asarray(intensity, np.float32).reshape(-1)
        if not _skip_checks:
            order = np.argsort(mz)
            self._mz, self._intensity = self._mz[order], self._intensity[order]
        self.retention_time = retention_time

    @property
    def mz(self) -> np.ndarray:
        return self._mz

    @property
    def intensity(self) -> np.ndarray:
        return self._intensity

    def round(
        self, decimals: int = 0, combine: str = "sum"
    ) -> "MsmsSpectrumJit":
        mz_round = np.round(self._mz, decimals=decimals)
        mz_unique = np.unique(mz_round)
        if len(mz_unique) == len(mz_round):
            self._mz = mz_unique
        # If peaks got merged by rounding the mass-to-charge ratios we need to
        # combine their intensities and annotations as well.
        else:
            intensity_unique = np.zeros_like(mz_unique, np.float32)
            i_orig = offset = 0
            for i_unique in range(len(mz_unique)):
                # Check whether subsequent mz values got merged.
                while (
                    i_orig + offset < len(mz_round)
                    and abs(mz_unique[i_unique] - mz_round[i_orig + offset])
                    <= 1e-06
                ):
                    offset += 1
                # Combine the corresponding intensities.
                intensity_unique[i_unique] = (
                    self._intensity[i_orig : i_orig + offset].sum()
                    if combine == "sum"
                    else self._intensity[i_orig : i_orig + offset].max()
                )
                i_orig += offset
                offset = 0
            self._mz, self._intensity = mz_unique, intensity_unique
        return self

    def set_mz_range(
        self, min_mz: Optional[float] = None, max_mz: Optional[float] = None
    ) -> "MsmsSpectrumJit":
        if min_mz is None and max_mz is None:
            return self
        else:
            if min_mz is None:
                min_mz = self._mz[0]
            if max_mz is None:
                max_mz = self._mz[-1]
            if max_mz < min_mz:
                min_mz, max_mz = max_mz, min_mz
        min_i, max_i = 0, len(self._mz)
        while min_i < len(self._mz) and self._mz[min_i] < min_mz:
            min_i += 1
        while max_i > 0 and self._mz[max_i - 1] > max_mz:
            max_i -= 1
        self._mz = self._mz[min_i:max_i]
        self._intensity = self._intensity[min_i:max_i]
        return self

    def remove_precursor_peak(
        self,
        fragment_tol_mass: float,
        fragment_tol_mode: str,
        isotope: int = 0,
    ) -> MsmsSpectrumJit:
        # TODO: This assumes [M+H]x charged ions.
        adduct_mass = 1.007825
        neutral_mass = (
            self.precursor_mz - adduct_mass
        ) * self.precursor_charge
        c_mass_diff = 1.003355
        remove_mz = [
            (neutral_mass + iso * c_mass_diff) / charge + adduct_mass
            for charge in range(self.precursor_charge, 0, -1)
            for iso in range(isotope + 1)
        ]
        mask = np.full_like(self._mz, True, np.bool_)
        mz_i = remove_i = 0
        while mz_i < len(self._mz) and remove_i < len(remove_mz):
            md = utils.mass_diff(
                self._mz[mz_i], remove_mz[remove_i], fragment_tol_mode == "Da"
            )
            if md < -fragment_tol_mass:
                mz_i += 1
            elif md > fragment_tol_mass:
                remove_i += 1
            else:
                mask[mz_i] = False
                mz_i += 1
        self._mz, self._intensity = self._mz[mask], self._intensity[mask]
        return self

    def filter_intensity(
        self, min_intensity: float = 0.0, max_num_peaks: Optional[int] = None
    ) -> "MsmsSpectrumJit":
        if max_num_peaks is None:
            max_num_peaks = len(self._intensity)

        intensity_idx = np.argsort(self._intensity)
        min_intensity *= self._intensity[intensity_idx[-1]]
        # Discard low-intensity noise peaks.
        start_i = 0
        for intens in self._intensity[intensity_idx]:
            if intens > min_intensity:
                break
            start_i += 1
        # Only retain at most the `max_num_peaks` most intense peaks.
        mask = np.full_like(self._intensity, False, np.bool_)
        mask[
            intensity_idx[max(start_i, len(intensity_idx) - max_num_peaks) :]
        ] = True
        self._mz, self._intensity = self._mz[mask], self._intensity[mask]
        return self

    def scale_intensity(
        self,
        scaling: Optional[str] = None,
        max_intensity: Optional[float] = None,
        degree: int = 2,
        base: int = 2,
        max_rank: Optional[int] = None,
    ) -> "MsmsSpectrumJit":
        if scaling == "root":
            self._intensity = np.power(self._intensity, 1 / degree).astype(
                np.float32
            )
        elif scaling == "log":
            self._intensity = (
                np.log1p(self._intensity) / np.log(base)
            ).astype(np.float32)
        elif scaling == "rank":
            if max_rank is None:
                max_rank = len(self._intensity)
            if max_rank < len(self._intensity):
                raise ValueError(
                    "`max_rank` should be greater than or equal to the number "
                    "of peaks in the spectrum. See `filter_intensity` to "
                    "reduce the number of peaks in the spectrum."
                )
            self._intensity = (
                max_rank - np.argsort(np.argsort(self._intensity)[::-1])
            ).astype(np.float32)
        if max_intensity is not None:
            self._intensity = (
                self._intensity * max_intensity / self._intensity.max()
            ).astype(np.float32)
        return self


class MsmsSpectrum:
    """
    Class representing a tandem mass spectrum.
    """

    def __init__(
        self,
        identifier: str,
        precursor_mz: float,
        precursor_charge: int,
        mz: Union[np.ndarray, Iterable],
        intensity: Union[np.ndarray, Iterable],
        retention_time: float = np.nan,
    ) -> None:
        """
        Instantiate a new `MsmsSpectrum` consisting of fragment peaks.

        Parameters
        ----------
        identifier : str
            Spectrum identifier. It is recommended to use a unique and
            interpretable identifier, such as a `Universal Spectrum Identifier
            (USI) <https://psidev.info/usi>`_ as defined by the Proteomics
            Standards Initiative.
        precursor_mz : float
            Precursor ion m/z.
        precursor_charge : int
            Precursor ion charge.
        mz : array_like
            M/z values of the fragment peaks.
        intensity : array_like
            Intensities of the corresponding fragment peaks in `mz`.
        retention_time : float, optional
            Retention time at which the spectrum was acquired (the default is
            np.nan, which indicates that retention time is
            unspecified/unknown).
        """
        self._inner = MsmsSpectrumJit(
            identifier,
            precursor_mz,
            precursor_charge,
            np.require(mz, np.float64, "W"),
            np.require(intensity, np.float32, "W"),
            retention_time if retention_time is not None else np.nan,
        )
        self.proforma, self._annotation = None, None

    def __getstate__(self):
        return {
            "identifier": self.identifier,
            "precursor_mz": self.precursor_mz,
            "precursor_charge": self.precursor_charge,
            "mz": self.mz,
            "intensity": self.intensity,
            "retention_time": self.retention_time,
            "proforma": self.proforma,
            "annotation": self.annotation,
        }

    def __setstate__(self, state):
        self._inner = MsmsSpectrumJit(
            state["identifier"],
            state["precursor_mz"],
            state["precursor_charge"],
            state["mz"],
            state["intensity"],
            state["retention_time"],
            True,
        )
        self.proforma = state["proforma"]
        self._annotation = state["annotation"]

    @classmethod
    def from_usi(
        cls, usi: str, backend: str = "aggregator", **kwargs
    ) -> "MsmsSpectrum":
        """
        Construct a spectrum from a public resource specified by its Universal
        Spectrum Identifier (USI).

        See the `Universal Spectrum Identifier (USI)
        <https://psidev.info/usi>`_ specification by the Proteomics Standards
        Initiative for more information on how to use USIs to refer to spectra.

        Besides official USIs, the USI extension to refer to MS/MS spectra from
        various metabolomics resources is also supported. This behavior can be
        explicitly requested by specifying the `"gnps"` backend.
        See `the corresponding preprint
        <https://doi.org/10.1101/2020.05.09.086066>`_ for more information.

        See the `Pyteomics documentation
        <https://pyteomics.readthedocs.io/en/latest/api/usi.html#pyteomics.usi.proxi>`_
        for details on how to use specific PROXI backends.

        Parameters
        ----------
        usi : str
            The USI from which to generate the spectrum.
        backend : str
            PROXI host backend (default: 'aggregator').
        kwargs
            Extra arguments to construct the spectrum that might be missing
            from the PROXI response (e.g. `precursor_mz` or `precursor_charge`)
            and the PROXI backend (see the `pyteomics.usi.proxi`
            documentation).

        Returns
        -------
        MsmsSpectrum
            The spectrum corresponding to the given USI

        Raises
        ------
        ValueError
            If the PROXI response does not contain information about the
            precursor m/z or precursor charge. Explicit precursor m/z and
            precursor charge values can be provided using the `precursor_mz`
            and `precursor_charge` keyword arguments respectively.
        """
        try:
            spectrum_dict = pyteomics.usi.proxi(
                urllib.parse.quote_plus(usi), backend, **kwargs
            )
        except KeyError as e:
            if "'value'" in str(e):
                raise ValueError(
                    f"Invalid USI response format from PROXI server for USI: {usi}. "
                    f"This appears to be an issue with the upstream PROXI API or pyteomics library. "
                    f"Original error: {e}"
                ) from e
            else:
                raise
        if "precursor_mz" not in kwargs:
            for attr in spectrum_dict["attributes"]:
                if attr["accession"] in (
                    "MS:1000827",
                    "MS:1000744",
                    "MS:1002234",
                ):
                    kwargs["precursor_mz"] = float(attr["value"])
                    break
            else:
                raise ValueError(
                    "Unknown precursor m/z from USI. Specify the precursor m/z"
                    " directly."
                )
        if "precursor_charge" not in kwargs:
            for attr in spectrum_dict["attributes"]:
                if attr["accession"] == "MS:1000041":
                    kwargs["precursor_charge"] = int(attr["value"])
                    break
            else:
                raise ValueError(
                    "Unknown precursor charge from USI. Specify the precursor "
                    "charge directly."
                )
        mz = spectrum_dict["m/z array"]
        intensity = spectrum_dict["intensity array"]
        return cls(identifier=usi, mz=mz, intensity=intensity, **kwargs)

    @property
    def identifier(self) -> str:
        return self._inner.identifier

    @identifier.setter
    def identifier(self, identifier: str):
        self._inner.identifier = identifier

    @property
    def precursor_mz(self) -> float:
        return self._inner.precursor_mz

    @precursor_mz.setter
    def precursor_mz(self, precursor_mz: float):
        self._inner.precursor_mz = precursor_mz

    @property
    def precursor_charge(self) -> int:
        return self._inner.precursor_charge

    @precursor_charge.setter
    def precursor_charge(self, precursor_charge: int):
        self._inner.precursor_charge = precursor_charge

    @property
    def retention_time(self) -> float:
        return self._inner.retention_time

    @retention_time.setter
    def retention_time(self, retention_time: float):
        self._inner.retention_time = retention_time

    @property
    def mz(self) -> np.ndarray:
        """
        The m/z values of the fragment peaks.

        Returns
        -------
        np.ndarray
            The m/z values of the fragment peaks.
        """
        return self._inner.mz

    @property
    def intensity(self) -> np.ndarray:
        """
        TGet or set the intensities of the fragment peaks.

        When setting new intensity values it should be possible to convert the
        given values to a NumPy array and the number of intensity values should
        be equal to the number of m/z (and annotation) values.

        Returns
        -------
        np.ndarray
            The intensity values of the fragment peaks.
        """
        return self._inner.intensity

    @property
    def annotation(self) -> Optional[np.ndarray]:
        """
        The annotated labels of the fragment peaks.

        Returns
        -------
        Optional[np.ndarray]
            The labels of the fragment peaks, or None if the spectrum has not
            been annotated.
        """
        return self._annotation

    def round(self, decimals: int = 0, combine: str = "sum") -> "MsmsSpectrum":
        """
        Round the m/z values of the fragment peaks to the given number of
        decimals.

        Intensities of peaks that have the same m/z value after rounding will
        be combined using the specified strategy.

        Note: This peak manipulation will reset any ProForma annotations that
        were applied previously.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places to round the `mz` to (default: 0).
            If decimals is negative, it specifies the number of positions to
            the left of the decimal point.
        combine : {'sum', 'max'}
            Method used to combine intensities from merged fragment peaks.
            ``sum`` specifies that the intensities of the merged fragment peaks
            will be summed, ``max`` specifies that the maximum intensity of
            the fragment peaks that are merged is used (the default is
            ``sum``).

        Returns
        -------
        MsmsSpectrum
        """
        self.proforma, self._annotation = None, None
        self._inner.round(decimals, combine)
        return self

    def set_mz_range(
        self, min_mz: Optional[float] = None, max_mz: Optional[float] = None
    ) -> "MsmsSpectrum":
        """
        Restrict the m/z values of the fragment peaks to the given range.

        Note: This peak manipulation will reset any ProForma annotations that
        were applied previously.

        Parameters
        ----------
        min_mz : Optional[float], optional
            Minimum m/z (inclusive). If not set no minimal m/z restriction will
            occur.
        max_mz : Optional[float], optional
            Maximum m/z (inclusive). If not set no maximal m/z restriction will
            occur.

        Returns
        -------
        MsmsSpectrum
        """
        self.proforma, self._annotation = None, None
        self._inner.set_mz_range(min_mz, max_mz)
        return self

    def remove_precursor_peak(
        self,
        fragment_tol_mass: float,
        fragment_tol_mode: str,
        isotope: int = 0,
    ) -> "MsmsSpectrum":
        """
        Remove fragment peak(s) close to the precursor m/z.

        Note: This peak manipulation will reset any ProForma annotations
        that were applied previously.

        Parameters
        ----------
        fragment_tol_mass : float
            Fragment mass tolerance around the precursor mass to remove
            the precursor peak.
        fragment_tol_mode : {'Da', 'ppm'}
            Fragment mass tolerance unit. Either 'Da' or 'ppm'.
        isotope : int
            The number of precursor isotopic peaks to be checked (the
            default is 0 to check only the mono-isotopic peaks).

        Returns
        -------
        MsmsSpectrum
        """
        if fragment_tol_mode not in ("Da", "ppm"):
            raise ValueError(
                "Unknown fragment mass tolerance unit specified. Supported "
                'values are "Da" or "ppm".'
            )
        self.proforma, self._annotation = None, None
        self._inner.remove_precursor_peak(
            fragment_tol_mass, fragment_tol_mode, isotope
        )
        return self

    def filter_intensity(
        self, min_intensity: float = 0.0, max_num_peaks: Optional[int] = None
    ) -> "MsmsSpectrum":
        """
        Remove low-intensity fragment peaks.

        Only the `max_num_peaks` most intense fragment peaks are retained.
        Additionally, noise peaks whose intensity is below `min_intensity`
        percentage of the intensity of the most intense peak are removed.

        Note: This peak manipulation will reset any ProForma annotations that
        were applied previously.

        Parameters
        ----------
        min_intensity : float, optional
            Remove peaks whose intensity is below `min_intensity` percentage
            of the intensity of the most intense peak (the default is 0, which
            means that no minimum intensity filter will be applied).
        max_num_peaks : Optional[int], optional
            Only retain the `max_num_peaks` most intense peaks (the default is
            None, which retains all peaks).

        Returns
        -------
        MsmsSpectrum
        """
        self.proforma, self._annotation = None, None
        self._inner.filter_intensity(min_intensity, max_num_peaks)
        return self

    def scale_intensity(
        self,
        scaling: Optional[str] = None,
        max_intensity: Optional[float] = None,
        *,
        degree: int = 2,
        base: int = 2,
        max_rank: Optional[int] = None,
    ) -> "MsmsSpectrum":
        """
        Scale the intensity of the fragment peaks.

        Two types of scaling can be performed: scaling all peaks using a
        specific transformation and scaling the peaks relative to the most
        intense peak.

        Parameters
        ----------
        scaling : {'root', 'log', 'rank'}, optional
            Method to scale the peak intensities (the default is None, which
            means that no transformation will be performed).
            Potential transformation options are:

            - 'root': Root-transform the peak intensities. The default is a
              square root transformation (`degree` is 2). The degree of the
              root can be specified using the `degree` kwarg.
            - 'log':  Log-transform the peak intensities. The default is a log2
              transformation (`base` is 2) after summing the intensities with 1
              to avoid negative values after the transformation. The base of
              the logarithm can be specified using the `base` kwarg.
            - 'rank': Rank-transform the peak intensities. The maximum rank of
              the most intense peak can be specified using the `max_rank`
              kwarg, by default the number of peaks in the spectrum is used as
              the maximum rank.

        degree: int
            The degree of the root transformation (the default is 2 for square
            root scaling).
        base: int
            The base of the log transformation (the default is 2 for log2
            scaling).
        max_rank: Optional[int]
            The maximum rank of the rank transformation (the default is None,
            which uses the number of peaks in the spectrum as maximum rank).
            `max_rank` should be greater than or equal to the number of peaks
            in the spectrum.
        max_intensity : Optional[float], optional
            Intensity of the most intense peak relative to which the peaks will
            be scaled (the default is None, which means that no scaling
            relative to the most intense peak will be performed).

        Returns
        -------
        MsmsSpectrum
        """
        self._inner.scale_intensity(
            scaling, max_intensity, degree, base, max_rank
        )
        return self

    def _annotate_proteoforms(
        self,
        proteoforms: list[proforma.Proteoform],
        proforma_str: str,
        fragment_tol_mass: float,
        fragment_tol_mode: str,
        ion_types: str = "by",
        *,
        max_isotope: int = 0,
        max_ion_charge: Optional[int] = None,
        neutral_losses: Union[bool, Dict[Optional[str], float]] = False,
    ) -> MsmsSpectrum:
        """
        Assign fragment ion labels to the peaks from a ProForma
        annotation.

        This is meant to be an internal function that uses a pre-parsed
        ProForma string instead of parsing it internally. This can be
        useful when the same parsed sequence is used multiple times
        (since parsing the sequence is a lot slower than annotating the
        peaks).

        >>> proforma_sequence = "MYPEPTIDEK/2"
        >>> spectrum.annotate_proforma(proforma_sequence, ...)

        or

        >>> parsed_proforma = proforma.parse(proforma_sequence)
        >>> spectrum._annotate_proteoforms(parsed_proforma, proforma_sequence, ...)

        WARN:
            This function does not check that the passed sequence
            corresponds to the passed proteoforms.

        For additional information on the arguments, see the
        `MsmsSpectrum.annotate_proforma` documentation.
        """
        if fragment_tol_mode not in ("Da", "ppm"):
            raise ValueError(
                "Unknown fragment mass tolerance unit specified. Supported "
                'values are "Da" or "ppm".'
            )
        self.proforma = proforma_str
        mass_diff = functools.partial(
            utils.mass_diff, mode_is_da=fragment_tol_mode == "Da"
        )

        self._annotation = np.full_like(self.mz, None, object)
        # By default, peak charges are assumed to be smaller than the
        # precursor charge.
        if max_ion_charge is None:
            max_ion_charge = max(1, self.precursor_charge - 1)
        # Make sure the standard peaks (without a neutral loss) are
        # always considered.
        if isinstance(neutral_losses, bool):
            if not neutral_losses:
                neutral_losses = {None: 0}
            else:
                neutral_losses = fa.NEUTRAL_LOSS
        if neutral_losses is not None and None not in neutral_losses:
            neutral_losses[None] = 0

        analyte_number = 1 if len(proteoforms) > 1 else None
        for proteoform in proteoforms:
            fragments = fa.get_theoretical_fragments(
                proteoform,
                ion_types,
                max_isotope=max_isotope,
                max_charge=max_ion_charge,
                neutral_losses=neutral_losses,
            )
            fragment_i = 0
            for peak_i, peak_mz in enumerate(self.mz):
                pi = fa.PeakInterpretation()
                while (
                    fragment_i < len(fragments)
                    and mass_diff(peak_mz, fragments[fragment_i][1])
                    > fragment_tol_mass
                ):
                    fragment_i += 1
                i = 0
                while (
                    fragment_i + i < len(fragments)
                    and abs(mass_diff(peak_mz, fragments[fragment_i + i][1]))
                    <= fragment_tol_mass
                ):
                    # FIXME: Annotations should not be duplicated across
                    #   multiple peaks.
                    fragment = copy.copy(fragments[fragment_i + i][0])
                    fragment.analyte_number = analyte_number
                    fragment.mz_delta = (
                        round(
                            mass_diff(peak_mz, fragments[fragment_i + i][1]),
                            ndigits=5 if fragment_tol_mode == "Da" else 1,
                        ),
                        fragment_tol_mode,
                    )
                    pi.fragment_annotations.append(fragment)
                    i += 1
                self.annotation[peak_i] = pi
            if analyte_number is not None:
                analyte_number += 1

        return self

    def annotate_proforma(
        self,
        proforma_str: str,
        fragment_tol_mass: float,
        fragment_tol_mode: str,
        ion_types: str = "by",
        *,
        max_isotope: int = 0,
        max_ion_charge: Optional[int] = None,
        neutral_losses: Union[bool, Dict[Optional[str], float]] = False,
    ) -> MsmsSpectrum:
        """
        Assign fragment ion labels to the peaks from a ProForma
        annotation.

        Parameters
        ----------
        proforma_str : str
            The ProForma spectrum annotation.
        fragment_tol_mass : float
            Fragment mass tolerance to match spectrum peaks against
            theoretical peaks.
        fragment_tol_mode : {'Da', 'ppm'}
            Fragment mass tolerance unit. Either 'Da' or 'ppm'.
        ion_types : str, optional
            The ion types to generate. Can be any combination of 'a',
            'b', 'c', 'x', 'y', and 'z' for peptide fragments, 'I' for
            immonium ions, 'm' for internal fragment ions, 'p' for the
            precursor ion, and 'r' for reporter ions. The default is
            'by', which means that b and y peptide ions will be
            generated.
        max_isotope : int
            The maximum isotope to consider for the fragment ions (the
            default is 0 to consider only monoisotopic peaks).
        max_ion_charge : Optional[int], optional
            All fragments up to and including the given charge will be
            annotated (by default all fragments with a charge up to the
            precursor minus one (minimum charge one) will be annotated).
        neutral_losses : Union[bool, Dict[Optional[str], float]], optional
            Neutral losses to consider for each peak. If `None` or
            `False`, no neutral losses are considered. If specified as a
            dictionary, keys should be the molecular formulas of the
            neutral losses and values the corresponding mass
            differences. Note that mass differences should typically be
            negative. If `True`, all of the following neutral losses are
            considered:

            - Loss of hydrogen (H): -1.007825.
            - Loss of ammonia (NH3): -17.026549.
            - Loss of water (H2O): -18.010565.
            - Loss of carbon monoxide (CO): -27.994915.
            - Loss of carbon dioxide (CO2): -43.989829.
            - Loss of formamide (HCONH2): -45.021464.
            - Loss of formic acid (HCOOH): -46.005479.
            - Loss of methanesulfenic acid (CH4OS): -63.998301.
            - Loss of sulfur trioxide (SO3): -79.956818.
            - Loss of metaphosphoric acid (HPO3): -79.966331.
            - Loss of mercaptoacetamide (C2H5NOS): -91.009195.
            - Loss of mercaptoacetic acid (C2H4O2S): -91.993211.
            - Loss of phosphoric acid (H3PO4): -97.976896.

        Returns
        -------
        MsmsSpectrum
        """
        proteoforms = proforma.parse(proforma_str)

        return self._annotate_proteoforms(
            proforma_str=proforma_str,
            proteoforms=proteoforms,
            fragment_tol_mass=fragment_tol_mass,
            fragment_tol_mode=fragment_tol_mode,
            ion_types=ion_types,
            max_isotope=max_isotope,
            max_ion_charge=max_ion_charge,
            neutral_losses=neutral_losses,
        )
