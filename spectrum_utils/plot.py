import functools
import itertools
import math
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import spectrum_utils.fragment_annotation as fa
from spectrum_utils.spectrum import MsmsSpectrum
from spectrum_utils.utils import da_to_ppm, ppm_to_da


colors = {
    "a": "#388E3C",
    "b": "#1976D2",
    "c": "#00796B",
    "x": "#7B1FA2",
    "y": "#D32F2F",
    "z": "#F57C00",
    "m": "#FBC02D",
    "I": "#455A64",
    "p": "#512DA8",
    "?": "#212121",
    "f": "#212121",
    None: "#212121",
}
zorders = {
    "a": 3,
    "b": 4,
    "c": 3,
    "x": 3,
    "y": 4,
    "z": 3,
    "m": 2,
    "I": 3,
    "p": 3,
    "?": 2,
    "f": 5,
    None: 1,
}


def _format_ax(
    ax: plt.Axes,
    grid: Union[bool, str],
):
    """Set ax formatting options that are common to all plot types."""
    ax.xaxis.set_minor_locator(mticker.AutoLocator())
    ax.yaxis.set_minor_locator(mticker.AutoLocator())
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    if grid in (True, "both", "major"):
        ax.grid(True, "major", color="#9E9E9E", linewidth=0.2)
    if grid in (True, "both", "minor"):
        ax.grid(True, "minor", color="#9E9E9E", linewidth=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", labelsize="small")
    ax.set_xlabel("m/z", style="italic")


def _get_xlim(spec: MsmsSpectrum) -> Tuple[float, float]:
    """Get plot x-axis limits for a given spectrum."""
    round_mz = 50
    max_mz = math.ceil(spec.mz[-1] / round_mz + 1) * round_mz
    return 0.0, max_mz


def _annotate_ion(
    mz: float,
    intensity: float,
    annotation: Optional[fa.FragmentAnnotation],
    color_ions: bool,
    annot_fmt: Optional[Callable],
    annot_kws: Dict[str, object],
    ax: plt.Axes,
) -> Tuple[str, int]:
    """
    Annotate a specific fragment peak.

    Parameters
    ----------
    mz : float
        The peak's m/z value (position of the annotation on the x axis).
    intensity : float
        The peak's intensity (position of the annotation on the y axis).
    annotation : Optional[fa.FragmentAnnotation]
        The annotation that will be plotted.
    color_ions : bool
        Flag whether to color the peak annotation or not.
    annot_fmt : Optional[Callable]
        Function to format the peak annotations. See `FragmentAnnotation` for
        supported elements. By default, only canonical b and y peptide fragments
        are annotated. If `None`, no peaks are annotated.
    annot_kws : Dict[str, object]
        Keyword arguments for `ax.text` to customize peak annotations.
    ax : plt.Axes
        Axes instance on which to plot the annotation.

    Returns
    -------
    Tuple[str, int]
        A tuple of the annotation's color as a hex string and the annotation's
        zorder.
    """
    ion_type = annotation.ion_type[0] if annotation is not None else None
    color = colors.get(ion_type if color_ions else None)
    zorder = zorders.get(ion_type)
    if annot_fmt is not None and annotation is not None:
        y = intensity + 0.02 * (intensity > 0)
        kws = annot_kws.copy()
        kws.update(dict(color=color, zorder=zorder))
        ax.text(mz, y, annot_fmt(annotation), **kws)
    return color, zorder


def annotate_ion_type(
    annotation: fa.FragmentAnnotation, ion_types: Iterable[str]
) -> str:
    """
    Convert a `FragmentAnnotation` to a string for annotating peaks in a
    spectrum plot.

    This function will only annotate singly-charged, mono-isotopic canonical
    peaks with the given ion type(s).

    Parameters
    ----------
    annotation : fa.FragmentAnnotation
        The peak's fragment annotation.
    ion_types : Iterable[str]
        Accepted ion types to annotate.

    Returns
    -------
    str
        The peak's annotation string.
    """
    if (
        annotation.ion_type[0] in ion_types
        and annotation.neutral_loss is None
        and annotation.isotope == 0
        and annotation.charge == 1
    ):
        return str(annotation.ion_type)
    else:
        return ""


def spectrum(
    spec: MsmsSpectrum,
    *,
    color_ions: bool = True,
    annot_fmt: Optional[Callable] = functools.partial(
        annotate_ion_type, ion_types="by"
    ),
    annot_kws: Optional[Dict] = None,
    mirror_intensity: bool = False,
    grid: Union[bool, str] = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot an MS/MS spectrum.

    Parameters
    ----------
    spec : MsmsSpectrum
        The spectrum to be plotted.
    color_ions : bool, optional
        Flag indicating whether or not to color annotated fragment ions. The
        default is True.
    annot_fmt : Optional[Callable]
        Function to format the peak annotations. See `FragmentAnnotation` for
        supported elements. By default, only canonical b and y peptide fragments
        are annotated. If `None`, no peaks are annotated.
    annot_kws : Optional[Dict], optional
        Keyword arguments for `ax.text` to customize peak annotations.
    mirror_intensity : bool, optional
        Flag indicating whether to flip the intensity axis or not.
    grid : Union[bool, str], optional
        Draw grid lines or not. Either a boolean to enable/disable both major
        and minor grid lines or 'major'/'minor' to enable major or minor grid
        lines respectively.
    ax : Optional[plt.Axes], optional
        Axes instance on which to plot the spectrum. If None the current Axes
        instance is used.

    Returns
    -------
    plt.Axes
        The matplotlib Axes instance on which the spectrum is plotted.
    """
    if ax is None:
        ax = plt.gca()

    _format_ax(ax, grid)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(*(0, 1.15) if not mirror_intensity else (-1.15, 0))
    ax.set_ylabel("Intensity")

    if len(spec.mz) == 0:
        return ax

    ax.set_xlim(*_get_xlim(spec))

    max_intensity = spec.intensity.max()
    annotations = (
        spec.annotation
        if spec.annotation is not None
        else itertools.repeat(None)
    )
    annotation_kws = {
        "horizontalalignment": "left" if not mirror_intensity else "right",
        "verticalalignment": "center",
        "rotation": 90,
        "rotation_mode": "anchor",
        "zorder": 5,
    }
    if annot_kws is not None:
        annotation_kws.update(annot_kws)
    for mz, intensity, annotation in zip(spec.mz, spec.intensity, annotations):
        peak_intensity = intensity / max_intensity
        if mirror_intensity:
            peak_intensity *= -1

        color, zorder = _annotate_ion(
            mz,
            peak_intensity,
            # Use the first annotation in case there are multiple options.
            annotation[0] if annotation is not None else None,
            color_ions,
            annot_fmt,
            annotation_kws,
            ax,
        )
        ax.plot([mz, mz], [0, peak_intensity], color=color, zorder=zorder)

    return ax


def mass_errors(
    spec: MsmsSpectrum,
    *,
    unit: Optional[str] = None,
    plot_unknown: bool = True,
    color_ions: bool = True,
    grid: Union[bool, str] = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot mass error bubble plot for a given spectrum.

    A mass error bubble plot shows the error between observed and theoretical
    mass (y-axis) in function of the **m/z** (x-axis) for each peak in the
    spectrum. The size of the bubble is proportional to the intensity of the
    peak.

    Parameters
    ----------
    spec : MsmsSpectrum
        The spectrum with mass errors to be plotted.
    unit : str, optional
        The unit of the mass errors, either 'ppm', 'Da', or None. If None,
        the unit that was used for spectrum annotation is used. The default is
        None.
    plot_unknown : bool, optional
        Flag indicating whether or not to plot mass errors for unknown peaks.
    color_ions : bool, optional
        Flag indicating whether or not to color dots for annotated fragment
        ions. The default is True.
    grid : Union[bool, str], optional
        Draw grid lines or not. Either a boolean to enable/disable both major
        and minor grid lines or 'major'/'minor' to enable major or minor grid
        lines respectively.
    ax : Optional[plt.Axes], optional
        Axes instance on which to plot the mass errors. If None the current
        Axes instance is used.

    Returns
    -------
    plt.Axes
        The matplotlib Axes instance on which the mass errors are plotted.

    Notes
    -----
    The mass error bubble plot was first introduced in Barsnes et al. (2010).

    References
    ----------
    Barsnes, H., Eidhammer, I. and Martens, L. (2010).
    FragmentationAnalyzer: An open-source tool to analyze MS/MS
    fragmentation data. PROTEOMICS, 10, 1087–1090.
    doi:10.1002/pmic.200900681

    """
    if ax is None:
        ax = plt.gca()

    _format_ax(ax, grid)

    if len(spec.mz) == 0:
        ax.set_ylabel("Mass error")
        ax.set_ylim(-1, 1)
        return ax

    annotations = (
        spec.annotation
        if spec.annotation is not None
        else itertools.repeat(None, len(spec.mz))
    )

    known_ions = []
    dot_colors = []
    mz_deltas = []
    mz_delta_units = []
    for ann in annotations:
        # Use the first annotation in case there are multiple options.
        ion_type = ann[0].ion_type[0] if ann is not None else None
        is_known_ion = ion_type is not None and ion_type != "?"
        known_ions.append(is_known_ion)
        dot_colors.append(colors.get(ion_type if color_ions else None))
        mz_deltas.append(ann[0].mz_delta[0] if is_known_ion else 0.0)
        mz_delta_units.append(ann[0].mz_delta[1] if is_known_ion else None)

    dot_colors = np.array(dot_colors)
    mz_deltas = np.array(mz_deltas)
    intensity_scaled = 500 * (spec.intensity / np.max(spec.intensity))
    mask = (
        np.ones_like(spec.mz, dtype=bool)
        if plot_unknown
        else np.array(known_ions)
    )

    for known_unit in ["ppm", "Da"]:
        # Use `not any` instead of `all` to fail fast
        if not any(u and u != known_unit for u in mz_delta_units):
            annotation_unit = known_unit
            break
    else:
        raise ValueError("Inconsistent or unknown mass units in annotations.")
    if unit == "Da" and annotation_unit == "ppm":
        mz_deltas = ppm_to_da(mz_deltas, spec.mz)
    elif unit == "ppm" and annotation_unit == "Da":
        mz_deltas = da_to_ppm(mz_deltas, spec.mz)

    y_lim = 1.2 * np.max(np.abs(mz_deltas))
    if y_lim > 0.0:
        ax.set_ylim(-y_lim, y_lim)
    ax.set_xlim(*_get_xlim(spec))
    ax.set_ylabel(f"Mass error ({unit or annotation_unit})")

    ax.scatter(
        spec.mz[mask],
        mz_deltas[mask],
        s=intensity_scaled[mask],
        c=dot_colors[mask],
        alpha=0.5,
        edgecolors="none",
    )

    return ax


def mirror(
    spec_top: MsmsSpectrum,
    spec_bottom: MsmsSpectrum,
    spectrum_kws: Optional[Dict] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Mirror plot two MS/MS spectra.

    Parameters
    ----------
    spec_top : MsmsSpectrum
        The spectrum to be plotted on the top.
    spec_bottom : MsmsSpectrum
        The spectrum to be plotted on the bottom.
    spectrum_kws : Optional[Dict], optional
        Keyword arguments for `plot.spectrum`.
    ax : Optional[plt.Axes], optional
        Axes instance on which to plot the spectrum. If None the current Axes
        instance is used.

    Returns
    -------
    plt.Axes
        The matplotlib Axes instance on which the spectra are plotted.
    """
    if ax is None:
        ax = plt.gca()

    if spectrum_kws is None:
        spectrum_kws = {}
    # Top spectrum.
    spectrum(spec_top, mirror_intensity=False, ax=ax, **spectrum_kws)
    y_max = ax.get_ylim()[1]
    # Mirrored bottom spectrum.
    spectrum(spec_bottom, mirror_intensity=True, ax=ax, **spectrum_kws)
    y_min = ax.get_ylim()[0]
    ax.set_ylim(y_min, y_max)

    ax.axhline(0, color="#9E9E9E", zorder=10)

    max_mz_top = spec_top.mz[-1] if len(spec_top.mz) > 0 else 1
    max_mz_bottom = spec_bottom.mz[-1] if len(spec_bottom.mz) > 0 else 1
    # Update axes so that both spectra fit.
    round_mz = 50
    max_mz = max(
        [
            math.ceil(max_mz_top / round_mz + 1) * round_mz,
            math.ceil(max_mz_bottom / round_mz + 1) * round_mz,
        ]
    )
    ax.set_xlim(0, max_mz)
    ax.yaxis.set_major_locator(mticker.AutoLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{abs(x):.0%}")
    )

    return ax


def facet(
    spec_top: MsmsSpectrum,
    spec_mass_errors: Optional[MsmsSpectrum] = None,
    spec_bottom: Optional[MsmsSpectrum] = None,
    spectrum_kws: Optional[Mapping[str, Any]] = None,
    mass_errors_kws: Optional[Mapping[str, Any]] = None,
    height: Optional[float] = None,
    width: Optional[float] = None,
) -> plt.Figure:
    """
    Plot a spectrum, and optionally mass errors, and a mirror spectrum.

    Parameters
    ----------
    spec_top : MsmsSpectrum
        The spectrum to be plotted on the top.
    spec_mass_errors : Optional[MsmsSpectrum], optional
        The spectrum for which mass errors are to be plotted in the middle.
    spec_bottom : Optional[MsmsSpectrum], optional
        The spectrum to be plotted on the bottom.
    spectrum_kws : Optional[Mapping[str, Any]], optional
        Keyword arguments for `plot.spectrum` for the top and bottom spectra.
    mass_errors_kws : Optional[Mapping[str, Any]], optional
        Keyword arguments for `plot.mass_errors`.
    height : Optional[float], optional
        The height of the figure in inches.
    width : Optional[float], optional
        The width of the figure in inches.

    Returns
    -------
    plt.Figure
        The matplotlib Figure instance on which the spectra and mass errors
        are plotted.
    """

    n_rows = 1 + (spec_mass_errors is not None) + (spec_bottom is not None)
    height_ratios = [1]
    if spec_mass_errors is not None:
        height_ratios.append(0.5)
    if spec_bottom is not None:
        height_ratios.append(1)

    fig, axes = plt.subplots(
        *(n_rows, 1),
        figsize=(width or 7.5, height or (3.75 if spec_bottom is None else 6)),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = np.array(axes).flatten()

    spectrum(spec_top, ax=axes[0], **spectrum_kws or {})

    if spec_mass_errors is not None:
        mass_errors(spec_mass_errors, ax=axes[1], **mass_errors_kws or {})
        axes[0].get_xaxis().get_label().set_visible(False)

    if spec_bottom is not None:
        spectrum(
            spec_bottom,
            mirror_intensity=True,
            ax=axes[-1],
            **spectrum_kws or {},
        )
        for ax in axes[:-1]:
            ax.get_xaxis().get_label().set_visible(False)

        axes[-1].yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, pos: f"{abs(x):.0%}")
        )

    fig.align_ylabels(axes)
    fig.tight_layout()

    return fig
