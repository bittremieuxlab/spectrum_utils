import numpy as np
import pytest

from spectrum_utils import fragment_annotation, proforma


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(13)


def test_fragment_annotation_unknown():
    fragment_annotation.FragmentAnnotation("?")
    with pytest.raises(ValueError):
        fragment_annotation.FragmentAnnotation("?", neutral_loss="-H2O")
    with pytest.raises(ValueError):
        fragment_annotation.FragmentAnnotation("?", isotope=1)
    with pytest.raises(ValueError):
        fragment_annotation.FragmentAnnotation("?", charge=1)
    with pytest.raises(ValueError):
        fragment_annotation.FragmentAnnotation("?", adduct="[M+H]")


def test_fragment_annotation_primary():
    fragment_annotation.FragmentAnnotation(
        "b5", neutral_loss="-H2O", isotope=1, charge=1, adduct="[M+H]"
    )
    with pytest.raises(ValueError):
        fragment_annotation.FragmentAnnotation("b5", charge=0)
    with pytest.raises(ValueError):
        fragment_annotation.FragmentAnnotation("b5", charge=-2)


@pytest.mark.parametrize(
    ("mz_delta", "expected"),
    [
        ((1.23456789, "ppm"), "b5/1.2ppm"),
        ((0.00123456789, "Da"), "b5/0.00123"),
    ],
)
def test_fragment_annotation_mz_delta_display_precision(mz_delta, expected):
    annotation = fragment_annotation.FragmentAnnotation(
        "b5", charge=1, mz_delta=mz_delta
    )
    assert str(annotation) == expected


def test_get_theoretical_fragments():
    peptide = proforma.parse("HPYLEDR")[0]
    fragments = {
        "b1^1": 138.066147,
        "b2^1": 235.118912,
        "b3^1": 398.182220,
        "b4^1": 511.266266,
        "b5^1": 640.308899,
        "b6^1": 755.335815,
        "y1^1": 175.118912,
        "y2^1": 290.145844,
        "y3^1": 419.188446,
        "y4^1": 532.272522,
        "y5^1": 695.335815,
        "y6^1": 792.388550,
        "b1^2": 69.536731,
        "b2^2": 118.063111,
        "b3^2": 199.594776,
        "b4^2": 256.136806,
        "b5^2": 320.658101,
        "b6^2": 378.171571,
        "y1^2": 88.063114,
        "y2^2": 145.576584,
        "y3^2": 210.097879,
        "y4^2": 266.639909,
        "y5^2": 348.171574,
        "y6^2": 396.697954,
        "b1^3": 46.693580,
        "b2^3": 79.044500,
        "b3^3": 133.398943,
        "b4^3": 171.093630,
        "b5^3": 214.107826,
        "b6^3": 252.450140,
        "y1^3": 59.044501,
        "y2^3": 97.386815,
        "y3^3": 140.401011,
        "y4^3": 178.095698,
        "y5^3": 232.450141,
        "y6^3": 264.801061,
    }
    for (
        annotation,
        fragment_mz,
    ) in fragment_annotation.get_theoretical_fragments(peptide, max_charge=3):
        assert fragment_mz == pytest.approx(
            fragments[f"{annotation.ion_type}^{annotation.charge}"]
        )


def test_get_theoretical_fragments_static_mod():
    peptide = proforma.parse("<[+79.96633]@Y>HPYLEDR")[0]
    fragments = {
        "b1^1": 138.066147,
        "b2^1": 235.118912,
        "b3^1": 478.148590,
        "b4^1": 591.232666,
        "b5^1": 720.275269,
        "b6^1": 835.302185,
        "y1^1": 175.118912,
        "y2^1": 290.145844,
        "y3^1": 419.188446,
        "y4^1": 532.272522,
        "y5^1": 775.302185,
        "y6^1": 872.354980,
        "b1^2": 69.536731,
        "b2^2": 118.063111,
        "b3^2": 239.577941,
        "b4^2": 296.119971,
        "b5^2": 360.641266,
        "b6^2": 418.154736,
        "y1^2": 88.063114,
        "y2^2": 145.576584,
        "y3^2": 210.097879,
        "y4^2": 266.639909,
        "y5^2": 388.154739,
        "y6^2": 436.681119,
        "b1^3": 46.693580,
        "b2^3": 79.044500,
        "b3^3": 160.054386,
        "b4^3": 197.749073,
        "b5^3": 240.763270,
        "b6^3": 279.105583,
        "y1^3": 59.044501,
        "y2^3": 97.386815,
        "y3^3": 140.401011,
        "y4^3": 178.095698,
        "y5^3": 259.105585,
        "y6^3": 291.456505,
    }
    for (
        annotation,
        fragment_mz,
    ) in fragment_annotation.get_theoretical_fragments(peptide, max_charge=3):
        assert fragment_mz == pytest.approx(
            fragments[f"{annotation.ion_type}^{annotation.charge}"]
        )


def test_get_theoretical_fragments_mod():
    peptide = proforma.parse("HPY[+79.96633]LEDR")[0]
    fragments = {
        "b1^1": 138.066147,
        "b2^1": 235.118912,
        "b3^1": 478.148590,
        "b4^1": 591.232666,
        "b5^1": 720.275269,
        "b6^1": 835.302185,
        "y1^1": 175.118912,
        "y2^1": 290.145844,
        "y3^1": 419.188446,
        "y4^1": 532.272522,
        "y5^1": 775.302185,
        "y6^1": 872.354980,
        "b1^2": 69.536731,
        "b2^2": 118.063111,
        "b3^2": 239.577941,
        "b4^2": 296.119971,
        "b5^2": 360.641266,
        "b6^2": 418.154736,
        "y1^2": 88.063114,
        "y2^2": 145.576584,
        "y3^2": 210.097879,
        "y4^2": 266.639909,
        "y5^2": 388.154739,
        "y6^2": 436.681119,
        "b1^3": 46.693580,
        "b2^3": 79.044500,
        "b3^3": 160.054386,
        "b4^3": 197.749073,
        "b5^3": 240.763270,
        "b6^3": 279.105583,
        "y1^3": 59.044501,
        "y2^3": 97.386815,
        "y3^3": 140.401011,
        "y4^3": 178.095698,
        "y5^3": 259.105585,
        "y6^3": 291.456505,
    }
    for (
        annotation,
        fragment_mz,
    ) in fragment_annotation.get_theoretical_fragments(peptide, max_charge=3):
        assert fragment_mz == pytest.approx(
            fragments[f"{annotation.ion_type}^{annotation.charge}"]
        )


def test_get_theoretical_fragments_mod_term():
    peptide = proforma.parse("[+42.01056]-HPYLEDR")[0]
    fragments = {
        "b1": 180.076706,
        "b2": 277.129486,
        "b3": 440.192810,
        "b4": 553.276917,
        "b5": 682.319519,
        "b6": 797.346436,
        "y1": 175.118912,
        "y2": 290.145844,
        "y3": 419.188446,
        "y4": 532.272522,
        "y5": 695.335815,
        "y6": 792.388550,
    }
    for (
        annotation,
        fragment_mz,
    ) in fragment_annotation.get_theoretical_fragments(peptide):
        assert fragment_mz == pytest.approx(
            fragments[f"{annotation.ion_type}"]
        )


def test_get_theoretical_fragments_mod_multiple():
    peptide = proforma.parse("[+42.01056]-HPY[+79.96633]LEDR")[0]
    fragments = {
        "b1": 180.076706,
        "b2": 277.129486,
        "b3": 520.159180,
        "b4": 633.243225,
        "b5": 762.285828,
        "b6": 877.312744,
        "y1": 175.118912,
        "y2": 290.145844,
        "y3": 419.188446,
        "y4": 532.272522,
        "y5": 775.302185,
        "y6": 872.354980,
    }
    for (
        annotation,
        fragment_mz,
    ) in fragment_annotation.get_theoretical_fragments(peptide):
        assert fragment_mz == pytest.approx(
            fragments[f"{annotation.ion_type}"]
        )


def test_get_theoretical_fragments_isotope():
    peptide = proforma.parse("HPYLEDR")[0]
    fragments = {
        "b1^1": 138.066147,
        "b2^1": 235.118912,
        "b3^1": 398.182220,
        "b4^1": 511.266266,
        "b5^1": 640.308899,
        "b6^1": 755.335815,
        "y1^1": 175.118912,
        "y2^1": 290.145844,
        "y3^1": 419.188446,
        "y4^1": 532.272522,
        "y5^1": 695.335815,
        "y6^1": 792.388550,
        "b1^2": 69.536731,
        "b2^2": 118.063111,
        "b3^2": 199.594776,
        "b4^2": 256.136806,
        "b5^2": 320.658101,
        "b6^2": 378.171571,
        "y1^2": 88.063114,
        "y2^2": 145.576584,
        "y3^2": 210.097879,
        "y4^2": 266.639909,
        "y5^2": 348.171574,
        "y6^2": 396.697954,
    }
    for num_isotopes in range(0, 3):
        annotations = fragment_annotation.get_theoretical_fragments(
            peptide, max_charge=2, max_isotope=num_isotopes
        )
        assert len(annotations) == len(fragments) * (num_isotopes + 1)
        for annotation, fragment_mz in annotations:
            assert fragment_mz == pytest.approx(
                fragments[f"{annotation.ion_type}^{annotation.charge}"]
                + 1.003_354 * annotation.isotope / annotation.charge
            )


def test_get_theoretical_fragments_neutral_loss():
    peptide = proforma.parse("HPYLEDR")[0]
    fragments = {
        "b1^1": 138.066147,
        "b2^1": 235.118912,
        "b3^1": 398.182220,
        "b4^1": 511.266266,
        "b5^1": 640.308899,
        "b6^1": 755.335815,
        "y1^1": 175.118912,
        "y2^1": 290.145844,
        "y3^1": 419.188446,
        "y4^1": 532.272522,
        "y5^1": 695.335815,
        "y6^1": 792.388550,
        "b1^2": 69.536731,
        "b2^2": 118.063111,
        "b3^2": 199.594776,
        "b4^2": 256.136806,
        "b5^2": 320.658101,
        "b6^2": 378.171571,
        "y1^2": 88.063114,
        "y2^2": 145.576584,
        "y3^2": 210.097879,
        "y4^2": 266.639909,
        "y5^2": 348.171574,
        "y6^2": 396.697954,
        "b1^3": 46.693580,
        "b2^3": 79.044500,
        "b3^3": 133.398943,
        "b4^3": 171.093630,
        "b5^3": 214.107826,
        "b6^3": 252.450140,
        "y1^3": 59.044501,
        "y2^3": 97.386815,
        "y3^3": 140.401011,
        "y4^3": 178.095698,
        "y5^3": 232.450141,
        "y6^3": 264.801061,
    }
    neutral_loss = "H2O", 18.010565  # water
    neutral_loss_fragments = {}
    for fragment, mz in fragments.items():
        charge = int(fragment.split("^")[1])
        fragment = f"{fragment}-{neutral_loss[0]}"
        neutral_loss_fragments[fragment] = mz - (neutral_loss[1] / charge)
    fragments = {**fragments, **neutral_loss_fragments}
    for (
        annotation,
        fragment_mz,
    ) in fragment_annotation.get_theoretical_fragments(
        peptide,
        max_charge=3,
        neutral_losses={None: 0, neutral_loss[0]: -neutral_loss[1]},
    ):
        assert fragment_mz == pytest.approx(
            fragments[
                f"""{annotation.ion_type}^{annotation.charge}{
                    annotation.neutral_loss
                    if annotation.neutral_loss is not None
                    else ""
                }"""
            ]
        )


def test_get_theoretical_fragments_mod_neutral_loss():
    peptide = proforma.parse("HPY[+79.96633]LEDR")[0]
    fragments = {
        "b1^1": 138.066147,
        "b2^1": 235.118912,
        "b3^1": 478.148590,
        "b4^1": 591.232666,
        "b5^1": 720.275269,
        "b6^1": 835.302185,
        "y1^1": 175.118912,
        "y2^1": 290.145844,
        "y3^1": 419.188446,
        "y4^1": 532.272522,
        "y5^1": 775.302185,
        "y6^1": 872.354980,
        "b1^2": 69.536731,
        "b2^2": 118.063111,
        "b3^2": 239.577941,
        "b4^2": 296.119971,
        "b5^2": 360.641266,
        "b6^2": 418.154736,
        "y1^2": 88.063114,
        "y2^2": 145.576584,
        "y3^2": 210.097879,
        "y4^2": 266.639909,
        "y5^2": 388.154739,
        "y6^2": 436.681119,
        "b1^3": 46.693580,
        "b2^3": 79.044500,
        "b3^3": 160.054386,
        "b4^3": 197.749073,
        "b5^3": 240.763270,
        "b6^3": 279.105583,
        "y1^3": 59.044501,
        "y2^3": 97.386815,
        "y3^3": 140.401011,
        "y4^3": 178.095698,
        "y5^3": 259.105585,
        "y6^3": 291.456505,
    }
    neutral_loss = "H2O", 18.010565  # water
    neutral_loss_fragments = {}
    for fragment, mz in fragments.items():
        charge = int(fragment.split("^")[1])
        fragment = f"{fragment}-{neutral_loss[0]}"
        neutral_loss_fragments[fragment] = mz - (neutral_loss[1] / charge)
    fragments = {**fragments, **neutral_loss_fragments}
    for (
        annotation,
        fragment_mz,
    ) in fragment_annotation.get_theoretical_fragments(
        peptide,
        max_charge=3,
        neutral_losses={None: 0, neutral_loss[0]: -neutral_loss[1]},
    ):
        assert fragment_mz == pytest.approx(
            fragments[
                f"""{annotation.ion_type}^{annotation.charge}{
                    annotation.neutral_loss
                    if annotation.neutral_loss is not None
                    else ""
                }"""
            ]
        )


def test_get_theoretical_fragments_ambiguous():
    with pytest.raises(ValueError):
        fragment_annotation.get_theoretical_fragments(
            proforma.parse("HPYLEBDR")[0]
        )
    with pytest.raises(ValueError):
        fragment_annotation.get_theoretical_fragments(
            proforma.parse("HPZYLEDR")[0]
        )


def test_get_theoretical_fragments_unsupported_ion_type():
    with pytest.raises(ValueError):
        fragment_annotation.get_theoretical_fragments(
            proforma.parse("HPYLEDR")[0], "l"
        )
