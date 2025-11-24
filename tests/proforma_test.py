import os
import shutil
import unittest.mock
from urllib.error import URLError

import lark
import pytest

from spectrum_utils import proforma


def test_proforma_unmodified():
    # Sequences containing standard amino acids.
    assert proforma.parse("PEPTIDE") == [
        proforma.Proteoform(sequence="PEPTIDE")
    ]
    assert proforma.parse("ACDEFGHIKL") == [
        proforma.Proteoform(sequence="ACDEFGHIKL")
    ]
    assert proforma.parse("MNPQRSTVWY") == [
        proforma.Proteoform(sequence="MNPQRSTVWY")
    ]
    # Sequences are case insensitive.
    assert proforma.parse("peptide") == [
        proforma.Proteoform(sequence="PEPTIDE")
    ]
    assert proforma.parse("acdefghikl") == [
        proforma.Proteoform(sequence="ACDEFGHIKL")
    ]
    assert proforma.parse("mnpqrstvwy") == [
        proforma.Proteoform(sequence="MNPQRSTVWY")
    ]
    # Sequences containing ambiguous amino acids.
    assert proforma.parse("PEPTJDE") == [
        proforma.Proteoform(sequence="PEPTJDE")
    ]
    assert proforma.parse("PEPTIOE") == [
        proforma.Proteoform(sequence="PEPTIOE")
    ]
    assert proforma.parse("PEUTIDE") == [
        proforma.Proteoform(sequence="PEUTIDE")
    ]
    assert proforma.parse("PEPTIXDE") == [
        proforma.Proteoform(sequence="PEPTIXDE")
    ]
    assert proforma.parse("BEPTIDE") == [
        proforma.Proteoform(sequence="BEPTIDE")
    ]
    assert proforma.parse("PEPTIZE") == [
        proforma.Proteoform(sequence="PEPTIZE")
    ]


# noinspection DuplicatedCode
# noinspection PyUnresolvedReferences
def test_proforma_name():
    # Unimod named modification without prefix.
    proteoform = proforma.parse("EM[Oxidation]EVEES[Phospho]PEK")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 15.994915
    assert proteoform.modifications[0].position == 1
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[0].source[0].name == "Oxidation"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.966331
    assert proteoform.modifications[1].position == 6
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[1].source[0].name == "Phospho"
    assert proteoform.modifications[1].label is None
    # Unimod named modification with prefix.
    proteoform = proforma.parse("EM[U:Oxidation]EVEES[U:Phospho]PEK")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 15.994915
    assert proteoform.modifications[0].position == 1
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[0].source[0].name == "Oxidation"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.966331
    assert proteoform.modifications[1].position == 6
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[1].source[0].name == "Phospho"
    assert proteoform.modifications[1].label is None
    # Unimod named modification with internal brackets in the name.
    proteoform = proforma.parse(
        "EM[Oxidation]EVE[Cation:Mg[II]]ES[Phospho]PEK"
    )[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 3
    )
    assert proteoform.modifications[0].mass == 15.994915
    assert proteoform.modifications[0].position == 1
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[0].source[0].name == "Oxidation"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 21.969392
    assert proteoform.modifications[1].position == 4
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:956"
    assert proteoform.modifications[1].source[0].name == "Cation:Mg[II]"
    assert proteoform.modifications[1].label is None
    assert proteoform.modifications[2].mass == 79.966331
    assert proteoform.modifications[2].position == 6
    assert (
        proteoform.modifications[2].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[2].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[2].source[0].name == "Phospho"
    assert proteoform.modifications[2].label is None
    # PSI-MOD named modification without prefix.
    proteoform = proforma.parse(
        "EM[L-methionine sulfoxide]EVEES[O-phospho-L-serine]PEK"
    )[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 15.994915
    assert proteoform.modifications[0].position == 1
    assert proteoform.modifications[0].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[0].source[0].accession == "MOD:00719"
    assert (
        proteoform.modifications[0].source[0].name == "L-methionine sulfoxide"
    )
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.966331
    assert proteoform.modifications[1].position == 6
    assert proteoform.modifications[1].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[1].source[0].accession == "MOD:00046"
    assert proteoform.modifications[1].source[0].name == "O-phospho-L-serine"
    assert proteoform.modifications[1].label is None
    # PSI-MOD named modification with prefix.
    proteoform = proforma.parse(
        "EM[M:L-methionine sulfoxide]EVEES[M:O-phospho-L-serine]PEK"
    )[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 15.994915
    assert proteoform.modifications[0].position == 1
    assert proteoform.modifications[0].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[0].source[0].accession == "MOD:00719"
    assert (
        proteoform.modifications[0].source[0].name == "L-methionine sulfoxide"
    )
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.966331
    assert proteoform.modifications[1].position == 6
    assert proteoform.modifications[1].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[1].source[0].accession == "MOD:00046"
    assert proteoform.modifications[1].source[0].name == "O-phospho-L-serine"
    assert proteoform.modifications[1].label is None
    # RESID named modification (mandatory prefix).
    proteoform = proforma.parse(
        "EM[R:L-methionine sulfone]EVEES[R:O-phospho-L-serine]PEK"
    )[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 31.989829
    assert proteoform.modifications[0].position == 1
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "RESID"
    )
    assert proteoform.modifications[0].source[0].accession == "RESID:AA0251"
    assert proteoform.modifications[0].source[0].name == "L-methionine sulfone"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.966331
    assert proteoform.modifications[1].position == 6
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "RESID"
    )
    assert proteoform.modifications[1].source[0].accession == "RESID:AA0037"
    assert proteoform.modifications[1].source[0].name == "O-phospho-L-serine"
    assert proteoform.modifications[1].label is None
    # XL-MOD named modification (mandatory prefix).
    try:
        proteoform = proforma.parse("EMEVTK[X:DSS#XL1]SESPEK")[0]
        assert proteoform.sequence == "EMEVTKSESPEK"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 1
        )
        assert proteoform.modifications[0].mass == 138.06807961
        assert proteoform.modifications[0].position == 5
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "XLMOD"
        )
        assert proteoform.modifications[0].source[0].accession == "XLMOD:02001"
        assert proteoform.modifications[0].source[0].name == "DSS"
        assert proteoform.modifications[0].label.type == proforma.LabelType.XL
        assert proteoform.modifications[0].label.label == "XL1"

    except SyntaxError as e:
        if "XLMOD" in str(e) and "OBO file" in str(e):
            pytest.skip(
                f"XLMOD parsing failed due to known upstream OBO format issue: {e}"
            )
        else:
            raise  # GNO named modification (mandatory prefix).
    proteoform = proforma.parse("NEEYN[G:G59626AS]K")[0]
    assert proteoform.sequence == "NEEYNK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 1931.69
    assert proteoform.modifications[0].position == 4
    assert proteoform.modifications[0].source[0].controlled_vocabulary == "GNO"
    assert proteoform.modifications[0].source[0].accession == "GNO:G59626AS"
    assert proteoform.modifications[0].source[0].name == "G59626AS"
    assert proteoform.modifications[0].label is None
    # Resolving incorrect terms. Parsing will succeed, but an error will be
    # thrown when lazy loading the modification.
    # Non-existing terms without prefix.
    proteoform = proforma.parse("EM[WeirdMod]EVEES[Phospho]PEK")[0]
    with pytest.raises(KeyError):
        print(proteoform.modifications[0].mass)
    # Non-existing terms with prefix.
    proteoform = proforma.parse("EM[U:WeirdMod]EVEES[U:Phospho]PEK")[0]
    with pytest.raises(KeyError):
        print(proteoform.modifications[0].mass)
    # Non-existing prefix (will be considered as no prefix).
    proteoform = proforma.parse("EM[RandomPrefix:Oxidation]EVEES[Phospho]PEK")[
        0
    ]
    with pytest.raises(KeyError):
        print(proteoform.modifications[0].mass)
    # Trigger resolving via accession (instead of mass in the preceding tests).
    proteoform = proforma.parse("EM[U:Oxidation]EVEES[U:Phospho]PEK")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[0].source[0].name == "Oxidation"
    assert proteoform.modifications[0].mass == 15.994915
    assert proteoform.modifications[0].position == 1
    assert proteoform.modifications[0].label is None


# noinspection DuplicatedCode
# noinspection PyUnresolvedReferences
def test_proforma_accession():
    # Unimod modification by accession.
    proteoform = proforma.parse("EM[UNIMOD:35]EVEES[UNIMOD:56]PEK")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 15.994915
    assert proteoform.modifications[0].position == 1
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[0].source[0].name == "Oxidation"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 45.029395
    assert proteoform.modifications[1].position == 6
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:56"
    assert proteoform.modifications[1].source[0].name == "Acetyl:2H(3)"
    assert proteoform.modifications[1].label is None
    # PSI-MOD modification by accession.
    proteoform = proforma.parse("EM[MOD:00719]EVEES[MOD:00046]PEK")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 15.994915
    assert proteoform.modifications[0].position == 1
    assert proteoform.modifications[0].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[0].source[0].accession == "MOD:00719"
    assert (
        proteoform.modifications[0].source[0].name == "L-methionine sulfoxide"
    )
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.966331
    assert proteoform.modifications[1].position == 6
    assert proteoform.modifications[1].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[1].source[0].accession == "MOD:00046"
    assert proteoform.modifications[1].source[0].name == "O-phospho-L-serine"
    assert proteoform.modifications[1].label is None
    # RESID modification by accession.
    proteoform = proforma.parse("EM[RESID:AA0581]EVEES[RESID:AA0037]PEK")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 15.994915
    assert proteoform.modifications[0].position == 1
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "RESID"
    )
    assert proteoform.modifications[0].source[0].accession == "RESID:AA0581"
    assert (
        proteoform.modifications[0].source[0].name
        == "L-methionine (R)-sulfoxide"
    )
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.966331
    assert proteoform.modifications[1].position == 6
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "RESID"
    )
    assert proteoform.modifications[1].source[0].accession == "RESID:AA0037"
    assert proteoform.modifications[1].source[0].name == "O-phospho-L-serine"
    assert proteoform.modifications[1].label is None
    # GNO modification by accession.
    proteoform = proforma.parse("NEEYN[GNO:G59626AS]K")[0]
    assert proteoform.sequence == "NEEYNK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 1931.69
    assert proteoform.modifications[0].position == 4
    assert proteoform.modifications[0].source[0].controlled_vocabulary == "GNO"
    assert proteoform.modifications[0].source[0].accession == "GNO:G59626AS"
    assert proteoform.modifications[0].source[0].name == "G59626AS"
    assert proteoform.modifications[0].label is None
    proteoform = proforma.parse(
        "YPVLN[GNO:G62765YT]VTMPN[GNO:G02815KT]NSNGKFDK"
    )[0]
    assert proteoform.sequence == "YPVLNVTMPNNSNGKFDK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 1720.59
    assert proteoform.modifications[0].position == 4
    assert proteoform.modifications[0].source[0].controlled_vocabulary == "GNO"
    assert proteoform.modifications[0].source[0].accession == "GNO:G62765YT"
    assert proteoform.modifications[0].source[0].name == "G62765YT"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 1234.43
    assert proteoform.modifications[1].position == 9
    assert proteoform.modifications[1].source[0].controlled_vocabulary == "GNO"
    assert proteoform.modifications[1].source[0].accession == "GNO:G02815KT"
    assert proteoform.modifications[1].source[0].name == "G02815KT"
    assert proteoform.modifications[1].label is None
    # Invalid accessions.
    proteoform = proforma.parse("EM[M:00719]EVEES[M:00046]PEK")[0]
    with pytest.raises(KeyError):
        print(proteoform.modifications[0].mass)
    proteoform = proforma.parse("EM[U:35]EVEES[U:56]PEK")[0]
    with pytest.raises(KeyError):
        print(proteoform.modifications[0].mass)
    proteoform = proforma.parse("EM[R:AA0581]EVEES[R:AA0037]PEK")[0]
    with pytest.raises(KeyError):
        print(proteoform.modifications[0].mass)
    proteoform = proforma.parse("EM[UNIMOD:one]EVEES[UNIMOD:two]PEK")[0]
    with pytest.raises(KeyError):
        print(proteoform.modifications[0].mass)
    proteoform = proforma.parse("EM[MOD:one]EVEES[MOD:two]PEK")[0]
    with pytest.raises(KeyError):
        print(proteoform.modifications[0].mass)
    proteoform = proforma.parse("EM[RESID:one]EVEES[RESID:two]PEK")[0]
    with pytest.raises(KeyError):
        print(proteoform.modifications[0].mass)
    try:
        proteoform = proforma.parse(
            "YPVLN[XLMOD:one]VTMPN[XLMOD:two]NSNGKFDK"
        )[0]
        with pytest.raises(KeyError):
            print(proteoform.modifications[0].mass)
    except SyntaxError as e:
        if "XLMOD" in str(e) and "OBO file" in str(e):
            pytest.skip(
                f"XLMOD parsing failed due to known upstream OBO format issue: {e}"
            )
        else:
            raise
    proteoform = proforma.parse("YPVLN[GNO:one]VTMPN[GNO:two]NSNGKFDK")[0]
    with pytest.raises(KeyError):
        print(proteoform.modifications[0].mass)


# noinspection DuplicatedCode
# noinspection PyUnresolvedReferences
def test_proforma_xlink():
    # DSS crosslink between two lysines.
    try:
        proteoform = proforma.parse("EMEVTK[XLMOD:02001#XL1]SESPEK[#XL1]")[0]
        assert proteoform.sequence == "EMEVTKSESPEK"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 2
        )
        assert proteoform.modifications[0].mass == 138.06807961
        assert proteoform.modifications[0].position == 5
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "XLMOD"
        )
        assert proteoform.modifications[0].source[0].accession == "XLMOD:02001"
        assert proteoform.modifications[0].source[0].name == "DSS"
        assert proteoform.modifications[0].label.type == proforma.LabelType.XL
        assert proteoform.modifications[0].label.label == "XL1"
        assert proteoform.modifications[1].mass is None
        assert proteoform.modifications[1].position == 11
        assert proteoform.modifications[1].source is None
        assert proteoform.modifications[1].label.type == proforma.LabelType.XL
        assert proteoform.modifications[1].label.label == "XL1"
        # DSS crosslink between two lysines and an EDC cross-link between two other
        # lysines.
        proteoform = proforma.parse(
            "EMK[XLMOD:02000#XL1]EVTKSE[XLMOD:02010#XL2]SK[#XL1]PEK[#XL2]AR"
        )[0]
        assert proteoform.sequence == "EMKEVTKSESKPEKAR"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 4
        )
        assert proteoform.modifications[0].mass == 138.06807961
        assert proteoform.modifications[0].position == 2
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "XLMOD"
        )
        assert proteoform.modifications[0].source[0].accession == "XLMOD:02000"
        assert proteoform.modifications[0].source[0].name == "BS3"
        assert proteoform.modifications[0].label.type == proforma.LabelType.XL
        assert proteoform.modifications[0].label.label == "XL1"
        assert proteoform.modifications[1].mass == -18.01056027
        assert proteoform.modifications[1].position == 8
        assert (
            proteoform.modifications[1].source[0].controlled_vocabulary
            == "XLMOD"
        )
        assert proteoform.modifications[1].source[0].accession == "XLMOD:02010"
        assert (
            proteoform.modifications[1].source[0].name
            == "1-ethyl-3-(3-Dimethylaminopropyl)carbodiimide hydrochloride"
        )
        assert proteoform.modifications[1].label.type == proforma.LabelType.XL
        assert proteoform.modifications[1].label.label == "XL2"
        assert proteoform.modifications[2].mass is None
        assert proteoform.modifications[2].position == 10
        assert proteoform.modifications[2].source is None
        assert proteoform.modifications[2].label.type == proforma.LabelType.XL
        assert proteoform.modifications[2].label.label == "XL1"
        assert proteoform.modifications[3].mass is None
        assert proteoform.modifications[3].position == 13
        assert proteoform.modifications[3].source is None
        assert proteoform.modifications[3].label.type == proforma.LabelType.XL
        assert proteoform.modifications[3].label.label == "XL2"
        # "Dead end" crosslink.
        proteoform = proforma.parse("EMEVTK[XLMOD:02001#XL1]SESPEK")[0]
        assert proteoform.sequence == "EMEVTKSESPEK"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 1
        )
        assert proteoform.modifications[0].mass == 138.06807961
        assert proteoform.modifications[0].position == 5
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "XLMOD"
        )
        assert proteoform.modifications[0].source[0].accession == "XLMOD:02001"
        assert proteoform.modifications[0].source[0].name == "DSS"
        assert proteoform.modifications[0].label.type == proforma.LabelType.XL
        assert proteoform.modifications[0].label.label == "XL1"
        proteoform = proforma.parse("EMEVTK[XLMOD:02001]SESPEK")[0]
        assert proteoform.sequence == "EMEVTKSESPEK"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 1
        )
        assert proteoform.modifications[0].mass == 138.06807961
        assert proteoform.modifications[0].position == 5
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "XLMOD"
        )
        assert proteoform.modifications[0].source[0].accession == "XLMOD:02001"
        assert proteoform.modifications[0].source[0].name == "DSS"
        assert proteoform.modifications[0].label is None
        # Inter-chain crosslink.
        proteoforms = proforma.parse(
            "SEK[XLMOD:02001#XL1]UENCE//EMEVTK[XLMOD:02001#XL1]SESPEK"
        )
        assert len(proteoforms) == 2
        proteoform = proteoforms[0]
        assert proteoform.sequence == "SEKUENCE"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 1
        )
        assert proteoform.modifications[0].mass == 138.06807961
        assert proteoform.modifications[0].position == 2
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "XLMOD"
        )
        assert proteoform.modifications[0].source[0].accession == "XLMOD:02001"
        assert proteoform.modifications[0].source[0].name == "DSS"
        assert proteoform.modifications[0].label.type == proforma.LabelType.XL
        assert proteoform.modifications[0].label.label == "XL1"
        proteoform = proteoforms[1]
        assert proteoform.sequence == "EMEVTKSESPEK"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 1
        )
        assert proteoform.modifications[0].mass == 138.06807961
        assert proteoform.modifications[0].position == 5
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "XLMOD"
        )
        assert proteoform.modifications[0].source[0].accession == "XLMOD:02001"
        assert proteoform.modifications[0].source[0].name == "DSS"
        assert proteoform.modifications[0].label.type == proforma.LabelType.XL
        assert proteoform.modifications[0].label.label == "XL1"
        proteoforms = proforma.parse(
            "SEK[XLMOD:02001#XL1]UENCE//EMEVTK[#XL1]SESPEK"
        )
        assert len(proteoforms) == 2
        proteoform = proteoforms[0]
        assert proteoform.sequence == "SEKUENCE"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 1
        )
        assert proteoform.modifications[0].mass == 138.06807961
        assert proteoform.modifications[0].position == 2
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "XLMOD"
        )
        assert proteoform.modifications[0].source[0].accession == "XLMOD:02001"
        assert proteoform.modifications[0].source[0].name == "DSS"
        assert proteoform.modifications[0].label.type == proforma.LabelType.XL
        assert proteoform.modifications[0].label.label == "XL1"
        proteoform = proteoforms[1]
        assert proteoform.sequence == "EMEVTKSESPEK"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 1
        )
        assert proteoform.modifications[0].mass is None
        assert proteoform.modifications[0].position == 5
        assert proteoform.modifications[0].source is None
        assert proteoform.modifications[0].label.type == proforma.LabelType.XL
        assert proteoform.modifications[0].label.label == "XL1"
        # Disulfide linkage.
        proteoform = proforma.parse("EVTSEKC[MOD:00034#XL1]LEMSC[#XL1]EFD")[0]
        assert proteoform.sequence == "EVTSEKCLEMSCEFD"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 2
        )
        assert proteoform.modifications[0].mass == -2.015650
        assert proteoform.modifications[0].position == 6
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "MOD"
        )
        assert proteoform.modifications[0].source[0].accession == "MOD:00034"
        assert (
            proteoform.modifications[0].source[0].name
            == "L-cystine (cross-link)"
        )
        assert proteoform.modifications[0].label.type == proforma.LabelType.XL
        assert proteoform.modifications[0].label.label == "XL1"
        assert proteoform.modifications[1].mass is None
        assert proteoform.modifications[1].position == 11
        assert proteoform.modifications[1].source is None
        assert proteoform.modifications[1].label.type == proforma.LabelType.XL
        assert proteoform.modifications[1].label.label == "XL1"
        proteoform = proforma.parse(
            "EVTSEKC[L-cystine (cross-link)#XL1]LEMSC[#XL1]EFD"
        )[0]
        assert proteoform.sequence == "EVTSEKCLEMSCEFD"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 2
        )
        assert proteoform.modifications[0].mass == -2.015650
        assert proteoform.modifications[0].position == 6
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "MOD"
        )
        assert proteoform.modifications[0].source[0].accession == "MOD:00034"
        assert (
            proteoform.modifications[0].source[0].name
            == "L-cystine (cross-link)"
        )
        assert proteoform.modifications[0].label.type == proforma.LabelType.XL
        assert proteoform.modifications[0].label.label == "XL1"
        assert proteoform.modifications[1].mass is None
        assert proteoform.modifications[1].position == 11
        assert proteoform.modifications[1].source is None
        assert proteoform.modifications[1].label.type == proforma.LabelType.XL
        assert proteoform.modifications[1].label.label == "XL1"
        proteoforms = proforma.parse(
            "FVNQHLC[MOD:00034#XL1]GSHLVEALYLVC[MOD:00034#XL2]GERGFFYTPKA//"
            "GIVEQC[MOD:00034#XL3]C[#XL1]TSIC[#XL3]SLYQLENYC[#XL2]N"
        )
        assert len(proteoforms) == 2
        proteoform = proteoforms[0]
        assert proteoform.sequence == "FVNQHLCGSHLVEALYLVCGERGFFYTPKA"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 2
        )
        assert proteoform.modifications[0].mass == -2.015650
        assert proteoform.modifications[0].position == 6
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "MOD"
        )
        assert proteoform.modifications[0].source[0].accession == "MOD:00034"
        assert (
            proteoform.modifications[0].source[0].name
            == "L-cystine (cross-link)"
        )
        assert proteoform.modifications[0].label.type == proforma.LabelType.XL
        assert proteoform.modifications[0].label.label == "XL1"
        assert proteoform.modifications[1].mass == -2.015650
        assert proteoform.modifications[1].position == 18
        assert (
            proteoform.modifications[1].source[0].controlled_vocabulary
            == "MOD"
        )
        assert proteoform.modifications[1].source[0].accession == "MOD:00034"
        assert (
            proteoform.modifications[1].source[0].name
            == "L-cystine (cross-link)"
        )
        assert proteoform.modifications[1].label.type == proforma.LabelType.XL
        assert proteoform.modifications[1].label.label == "XL2"
        proteoform = proteoforms[1]
        assert proteoform.sequence == "GIVEQCCTSICSLYQLENYCN"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 4
        )
        assert proteoform.modifications[0].mass == -2.015650
        assert proteoform.modifications[0].position == 5
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "MOD"
        )
        assert proteoform.modifications[0].source[0].accession == "MOD:00034"
        assert (
            proteoform.modifications[0].source[0].name
            == "L-cystine (cross-link)"
        )
        assert proteoform.modifications[0].label.type == proforma.LabelType.XL
        assert proteoform.modifications[0].label.label == "XL3"
        assert proteoform.modifications[1].mass is None
        assert proteoform.modifications[1].position == 6
        assert proteoform.modifications[1].source is None
        assert proteoform.modifications[1].label.type == proforma.LabelType.XL
        assert proteoform.modifications[1].label.label == "XL1"
        assert proteoform.modifications[2].mass is None
        assert proteoform.modifications[2].position == 10
        assert proteoform.modifications[2].source is None
        assert proteoform.modifications[2].label.type == proforma.LabelType.XL
        assert proteoform.modifications[2].label.label == "XL3"
        assert proteoform.modifications[3].mass is None
        assert proteoform.modifications[3].position == 19
        assert proteoform.modifications[3].source is None
        assert proteoform.modifications[3].label.type == proforma.LabelType.XL
        assert proteoform.modifications[3].label.label == "XL2"
        proteoform = proforma.parse("EVTSEKC[XLMOD:02009#XL1]LEMSC[#XL1]EFD")[
            0
        ]
        assert proteoform.sequence == "EVTSEKCLEMSCEFD"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 2
        )
        assert proteoform.modifications[0].mass == -2.01565007
        assert proteoform.modifications[0].position == 6
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "XLMOD"
        )
        assert proteoform.modifications[0].source[0].accession == "XLMOD:02009"
        assert proteoform.modifications[0].source[0].name == "Disulfide"
        assert proteoform.modifications[0].label.type == proforma.LabelType.XL
        assert proteoform.modifications[0].label.label == "XL1"
        assert proteoform.modifications[1].mass is None
        assert proteoform.modifications[1].position == 11
        assert proteoform.modifications[1].source is None
        assert proteoform.modifications[1].label.type == proforma.LabelType.XL
        assert proteoform.modifications[1].label.label == "XL1"
        proteoform = proforma.parse("EVTSEKC[X:Disulfide#XL1]LEMSC[#XL1]EFD")[
            0
        ]
        assert proteoform.sequence == "EVTSEKCLEMSCEFD"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 2
        )
        assert proteoform.modifications[0].mass == -2.01565007
        assert proteoform.modifications[0].position == 6
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "XLMOD"
        )
        assert proteoform.modifications[0].source[0].accession == "XLMOD:02009"
        assert proteoform.modifications[0].source[0].name == "Disulfide"
        assert proteoform.modifications[0].label.type == proforma.LabelType.XL
        assert proteoform.modifications[0].label.label == "XL1"
        assert proteoform.modifications[1].mass is None
        assert proteoform.modifications[1].position == 11
        assert proteoform.modifications[1].source is None
        assert proteoform.modifications[1].label.type == proforma.LabelType.XL
        assert proteoform.modifications[1].label.label == "XL1"
        proteoform = proforma.parse(
            "EVTSEKC[half cystine]LEMSC[half cystine]EFD"
        )[0]
        assert proteoform.sequence == "EVTSEKCLEMSCEFD"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 2
        )
        assert proteoform.modifications[0].mass == -1.007825
        assert proteoform.modifications[0].position == 6
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "MOD"
        )
        assert proteoform.modifications[0].source[0].accession == "MOD:00798"
        assert proteoform.modifications[0].source[0].name == "half cystine"
        assert proteoform.modifications[0].label is None
        assert proteoform.modifications[1].mass == -1.007825
        assert proteoform.modifications[1].position == 11
        assert (
            proteoform.modifications[1].source[0].controlled_vocabulary
            == "MOD"
        )
        assert proteoform.modifications[1].source[0].accession == "MOD:00798"
        assert proteoform.modifications[1].source[0].name == "half cystine"
        assert proteoform.modifications[1].label is None
        proteoform = proforma.parse(
            "EVTSEKC[MOD:00798]LEMSC[MOD:00798]EFDEVTSEKC[MOD:00798]LEMS"
            "C[MOD:00798]EFD"
        )[0]
        assert proteoform.sequence == "EVTSEKCLEMSCEFDEVTSEKCLEMSCEFD"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 4
        )
        assert proteoform.modifications[0].mass == -1.007825
        assert proteoform.modifications[0].position == 6
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "MOD"
        )
        assert proteoform.modifications[0].source[0].accession == "MOD:00798"
        assert proteoform.modifications[0].source[0].name == "half cystine"
        assert proteoform.modifications[0].label is None
        assert proteoform.modifications[1].mass == -1.007825
        assert proteoform.modifications[1].position == 11
        assert (
            proteoform.modifications[1].source[0].controlled_vocabulary
            == "MOD"
        )
        assert proteoform.modifications[1].source[0].accession == "MOD:00798"
        assert proteoform.modifications[1].source[0].name == "half cystine"
        assert proteoform.modifications[1].label is None
        assert proteoform.modifications[2].mass == -1.007825
        assert proteoform.modifications[2].position == 21
        assert (
            proteoform.modifications[2].source[0].controlled_vocabulary
            == "MOD"
        )
        assert proteoform.modifications[2].source[0].accession == "MOD:00798"
        assert proteoform.modifications[2].source[0].name == "half cystine"
        assert proteoform.modifications[2].label is None
        assert proteoform.modifications[3].mass == -1.007825
        assert proteoform.modifications[3].position == 26
        assert (
            proteoform.modifications[3].source[0].controlled_vocabulary
            == "MOD"
        )
        assert proteoform.modifications[3].source[0].accession == "MOD:00798"
        assert proteoform.modifications[3].source[0].name == "half cystine"
        assert proteoform.modifications[3].label is None
        proteoform = proforma.parse("EVTSEKC[UNIMOD:374#XL1]LEMSC[#XL1]EFD")[0]
        assert proteoform.sequence == "EVTSEKCLEMSCEFD"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 2
        )
        assert proteoform.modifications[0].mass == -1.007825
        assert proteoform.modifications[0].position == 6
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "UNIMOD"
        )
        assert proteoform.modifications[0].source[0].accession == "UNIMOD:374"
        assert proteoform.modifications[0].source[0].name == "Dehydro"
        assert proteoform.modifications[0].label.type == proforma.LabelType.XL
        assert proteoform.modifications[0].label.label == "XL1"
        assert proteoform.modifications[1].mass is None
        assert proteoform.modifications[1].position == 11
        assert proteoform.modifications[1].source is None
        assert proteoform.modifications[1].label.type == proforma.LabelType.XL
        assert proteoform.modifications[1].label.label == "XL1"

    except SyntaxError as e:
        if "XLMOD" in str(e) and "OBO file" in str(e):
            pytest.skip(
                f"XLMOD parsing failed due to known upstream OBO format issue: {e}"
            )
        else:
            raise


# noinspection PyUnresolvedReferences
def test_proforma_branch():
    proteoforms = proforma.parse("ETFGD[MOD:00093#BRANCH]//R[#BRANCH]ATER")
    proteoform = proteoforms[0]
    assert proteoform.sequence == "ETFGD"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == -0.984016
    assert proteoform.modifications[0].position == 4
    assert proteoform.modifications[0].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[0].source[0].accession == "MOD:00093"
    assert (
        proteoform.modifications[0].source[0].name == "L-aspartic acid 1-amide"
    )
    assert proteoform.modifications[0].label.type == proforma.LabelType.BRANCH
    assert proteoform.modifications[0].label.label == "BRANCH"
    proteoform = proteoforms[1]
    assert proteoform.sequence == "RATER"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass is None
    assert proteoform.modifications[0].position == 0
    assert proteoform.modifications[0].source is None
    assert proteoform.modifications[0].label.type == proforma.LabelType.BRANCH
    assert proteoform.modifications[0].label.label == "BRANCH"
    proteoforms = proforma.parse(
        "AVTKYTSSK[MOD:00134#BRANCH]//AGKQLEDGRTLSDYNIQKESTLHLVLRLRG-[#BRANCH]"
    )
    proteoform = proteoforms[0]
    assert proteoform.sequence == "AVTKYTSSK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == -18.010565
    assert proteoform.modifications[0].position == 8
    assert proteoform.modifications[0].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[0].source[0].accession == "MOD:00134"
    assert proteoform.modifications[0].source[0].name == "N6-glycyl-L-lysine"
    assert proteoform.modifications[0].label.type == proforma.LabelType.BRANCH
    assert proteoform.modifications[0].label.label == "BRANCH"
    proteoform = proteoforms[1]
    assert proteoform.sequence == "AGKQLEDGRTLSDYNIQKESTLHLVLRLRG"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass is None
    assert proteoform.modifications[0].position == "C-term"
    assert proteoform.modifications[0].source is None
    assert proteoform.modifications[0].label.type == proforma.LabelType.BRANCH
    assert proteoform.modifications[0].label.label == "BRANCH"


# noinspection PyUnresolvedReferences
def test_proforma_delta_mass():
    # Delta mass without prefix.
    proteoform = proforma.parse("EM[+15.9949]EVEES[+79.9663]PEK")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 15.9949
    assert proteoform.modifications[0].position == 1
    assert proteoform.modifications[0].source[0].mass == 15.9949
    assert proteoform.modifications[0].source[0].controlled_vocabulary is None
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.9663
    assert proteoform.modifications[1].position == 6
    assert proteoform.modifications[1].source[0].mass == 79.9663
    assert proteoform.modifications[1].source[0].controlled_vocabulary is None
    assert proteoform.modifications[1].label is None
    proteoform = proforma.parse("EM[+15.995]EVEES[-18.01]PEK")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 15.995
    assert proteoform.modifications[0].position == 1
    assert proteoform.modifications[0].source[0].mass == 15.995
    assert proteoform.modifications[0].source[0].controlled_vocabulary is None
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == -18.01
    assert proteoform.modifications[1].position == 6
    assert proteoform.modifications[1].source[0].mass == -18.01
    assert proteoform.modifications[1].source[0].controlled_vocabulary is None
    assert proteoform.modifications[1].label is None
    # Delta mass with CV prefix.
    proteoform = proforma.parse("EM[U:+15.9949]EVEES[U:+79.9663]PEK")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 15.9949
    assert proteoform.modifications[0].position == 1
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.9663
    assert proteoform.modifications[1].position == 6
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].label is None
    proteoform = proforma.parse("EM[U:+15.995]EVEES[U:+79.966]PEK")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 15.995
    assert proteoform.modifications[0].position == 1
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.966
    assert proteoform.modifications[1].position == 6
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].label is None
    # Delta mass with experimentally observed prefix.
    proteoform = proforma.parse("EM[U:+15.995]EVEES[Obs:+79.978]PEK")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 15.995
    assert proteoform.modifications[0].position == 1
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.978
    assert proteoform.modifications[1].position == 6
    assert proteoform.modifications[1].source[0].controlled_vocabulary is None
    assert proteoform.modifications[1].label is None


# noinspection PyUnresolvedReferences
def test_proforma_gap():
    proteoform = proforma.parse("RTAAX[+367.0537]WT")[0]
    assert proteoform.sequence == "RTAAXWT"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 367.0537
    assert proteoform.modifications[0].position == 4
    assert proteoform.modifications[0].source[0].mass == 367.0537
    assert proteoform.modifications[0].source[0].controlled_vocabulary is None
    assert proteoform.modifications[0].label is None


# noinspection PyUnresolvedReferences
def test_proforma_formula():
    proteoform = proforma.parse("SEQUEN[Formula:C12H20O2]CE")[0]
    assert proteoform.sequence == "SEQUENCE"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 196.14632988052
    assert proteoform.modifications[0].position == 5
    assert proteoform.modifications[0].source[0].formula == "C12H20O2"
    assert proteoform.modifications[0].source[0].isotopes is None
    assert proteoform.modifications[0].label is None
    proteoform = proforma.parse("SEQUEN[Formula:C12 H20 O2]CE")[0]
    assert proteoform.sequence == "SEQUENCE"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 196.14632988052
    assert proteoform.modifications[0].position == 5
    assert proteoform.modifications[0].source[0].formula == "C12 H20 O2"
    assert proteoform.modifications[0].source[0].isotopes is None
    assert proteoform.modifications[0].label is None
    proteoform = proforma.parse("SEQUEN[Formula:HN-1O2]CE")[0]
    assert proteoform.sequence == "SEQUENCE"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 18.99458026639
    assert proteoform.modifications[0].position == 5
    assert proteoform.modifications[0].source[0].formula == "HN-1O2"
    assert proteoform.modifications[0].source[0].isotopes is None
    assert proteoform.modifications[0].label is None
    # Mass calculation of isotopes is currently still unsupported.
    # FIXME: Add isotope support to Pyteomics.
    proteoform = proforma.parse("SEQUEN[Formula:[13C2]CH6N]CE")[0]
    assert proteoform.sequence == "SEQUENCE"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].position == 5
    assert proteoform.modifications[0].source[0].formula == "CH6N"
    assert proteoform.modifications[0].source[0].isotopes == ["13C2"]
    assert proteoform.modifications[0].label is None
    with pytest.raises(NotImplementedError):
        print(proteoform.modifications[0].mass)
    proteoform = proforma.parse("SEQUEN[Formula:[13C2][12C-2]H2N]CE")[0]
    assert proteoform.sequence == "SEQUENCE"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].position == 5
    assert proteoform.modifications[0].source[0].formula == "H2N"
    assert proteoform.modifications[0].source[0].isotopes == ["13C2", "12C-2"]
    assert proteoform.modifications[0].label is None
    with pytest.raises(NotImplementedError):
        print(proteoform.modifications[0].mass)
    proteoform = proforma.parse("SEQUEN[Formula:[13C2]C-2H2N]CE")[0]
    assert proteoform.sequence == "SEQUENCE"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].position == 5
    assert proteoform.modifications[0].source[0].formula == "C-2H2N"
    assert proteoform.modifications[0].source[0].isotopes == ["13C2"]
    assert proteoform.modifications[0].label is None
    with pytest.raises(NotImplementedError):
        print(proteoform.modifications[0].mass)


# noinspection DuplicatedCode
# noinspection PyUnresolvedReferences
def test_proforma_glycan():
    proteoform = proforma.parse("SEQUEN[Glycan:HexNAc1Hex2]CE")[0]
    assert proteoform.sequence == "SEQUENCE"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 527.185019356
    assert proteoform.modifications[0].position == 5
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications[0].source[0].composition) == 2
    )
    assert (
        proteoform.modifications[0].source[0].composition[0].monosaccharide
        == "HexNAc"
    )
    assert proteoform.modifications[0].source[0].composition[0].count == 1
    assert (
        proteoform.modifications[0].source[0].composition[1].monosaccharide
        == "Hex"
    )
    assert proteoform.modifications[0].source[0].composition[1].count == 2
    assert proteoform.modifications[0].label is None
    proteoform = proforma.parse("SEQUEN[Glycan:HexNAc1 Hex2]CE")[0]
    assert proteoform.sequence == "SEQUENCE"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 527.185019356
    assert proteoform.modifications[0].position == 5
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications[0].source[0].composition) == 2
    )
    assert (
        proteoform.modifications[0].source[0].composition[0].monosaccharide
        == "HexNAc"
    )
    assert proteoform.modifications[0].source[0].composition[0].count == 1
    assert (
        proteoform.modifications[0].source[0].composition[1].monosaccharide
        == "Hex"
    )
    assert proteoform.modifications[0].source[0].composition[1].count == 2
    assert proteoform.modifications[0].label is None


# noinspection DuplicatedCode
# noinspection PyUnresolvedReferences
def test_proforma_special():
    # N-terminal and C-terminal modifications.
    proteoform = proforma.parse("[iTRAQ4plex]-EM[Oxidation]EVNES[Phospho]PEK")[
        0
    ]
    assert proteoform.sequence == "EMEVNESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 3
    )
    assert proteoform.modifications[0].mass == 144.102063
    assert proteoform.modifications[0].position == "N-term"
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:214"
    assert proteoform.modifications[0].source[0].name == "iTRAQ4plex"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 15.994915
    assert proteoform.modifications[1].position == 1
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[1].source[0].name == "Oxidation"
    assert proteoform.modifications[1].label is None
    assert proteoform.modifications[2].mass == 79.966331
    assert proteoform.modifications[2].position == 6
    assert (
        proteoform.modifications[2].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[2].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[2].source[0].name == "Phospho"
    assert proteoform.modifications[2].label is None
    proteoform = proforma.parse(
        "[iTRAQ4plex]-EM[U:Oxidation]EVNES[Phospho]PEK[iTRAQ4plex]-[Methyl]"
    )[0]
    assert proteoform.sequence == "EMEVNESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 5
    )
    assert proteoform.modifications[0].mass == 144.102063
    assert proteoform.modifications[0].position == "N-term"
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:214"
    assert proteoform.modifications[0].source[0].name == "iTRAQ4plex"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 15.994915
    assert proteoform.modifications[1].position == 1
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[1].source[0].name == "Oxidation"
    assert proteoform.modifications[1].label is None
    assert proteoform.modifications[2].mass == 79.966331
    assert proteoform.modifications[2].position == 6
    assert (
        proteoform.modifications[2].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[2].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[2].source[0].name == "Phospho"
    assert proteoform.modifications[2].label is None
    assert proteoform.modifications[3].mass == 144.102063
    assert proteoform.modifications[3].position == 9
    assert (
        proteoform.modifications[3].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[3].source[0].accession == "UNIMOD:214"
    assert proteoform.modifications[3].source[0].name == "iTRAQ4plex"
    assert proteoform.modifications[3].label is None
    assert proteoform.modifications[4].mass == 14.01565
    assert proteoform.modifications[4].position == "C-term"
    assert (
        proteoform.modifications[4].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[4].source[0].accession == "UNIMOD:34"
    assert proteoform.modifications[4].source[0].name == "Methyl"
    assert proteoform.modifications[4].label is None
    # Labile modifications.
    proteoform = proforma.parse(
        "{Glycan:Hex}EM[U:Oxidation]EVNES[Phospho]PEK[iTRAQ4plex]"
    )[0]
    assert proteoform.sequence == "EMEVNESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 4
    )
    assert proteoform.modifications[0].mass == 162.052823418
    assert proteoform.modifications[0].position == "labile"
    assert (
        proteoform.modifications[0].source[0].composition[0].monosaccharide
        == "Hex"
    )
    assert proteoform.modifications[0].source[0].composition[0].count == 1
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 15.994915
    assert proteoform.modifications[1].position == 1
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[1].source[0].name == "Oxidation"
    assert proteoform.modifications[1].label is None
    assert proteoform.modifications[2].mass == 79.966331
    assert proteoform.modifications[2].position == 6
    assert (
        proteoform.modifications[2].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[2].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[2].source[0].name == "Phospho"
    assert proteoform.modifications[2].label is None
    assert proteoform.modifications[3].mass == 144.102063
    assert proteoform.modifications[3].position == 9
    assert (
        proteoform.modifications[3].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[3].source[0].accession == "UNIMOD:214"
    assert proteoform.modifications[3].source[0].name == "iTRAQ4plex"
    assert proteoform.modifications[3].label is None
    proteoform = proforma.parse(
        "{Glycan:Hex}[iTRAQ4plex]-EM[Oxidation]EVNES[Phospho]PEK[iTRAQ4plex]"
    )[0]
    assert proteoform.sequence == "EMEVNESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 5
    )
    assert proteoform.modifications[0].mass == 162.052823418
    assert proteoform.modifications[0].position == "labile"
    assert (
        proteoform.modifications[0].source[0].composition[0].monosaccharide
        == "Hex"
    )
    assert proteoform.modifications[0].source[0].composition[0].count == 1
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 144.102063
    assert proteoform.modifications[1].position == "N-term"
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:214"
    assert proteoform.modifications[1].source[0].name == "iTRAQ4plex"
    assert proteoform.modifications[1].label is None
    assert proteoform.modifications[2].mass == 15.994915
    assert proteoform.modifications[2].position == 1
    assert (
        proteoform.modifications[2].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[2].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[2].source[0].name == "Oxidation"
    assert proteoform.modifications[2].label is None
    assert proteoform.modifications[3].mass == 79.966331
    assert proteoform.modifications[3].position == 6
    assert (
        proteoform.modifications[3].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[3].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[3].source[0].name == "Phospho"
    assert proteoform.modifications[3].label is None
    assert proteoform.modifications[4].mass == 144.102063
    assert proteoform.modifications[4].position == 9
    assert (
        proteoform.modifications[4].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[4].source[0].accession == "UNIMOD:214"
    assert proteoform.modifications[4].source[0].name == "iTRAQ4plex"
    assert proteoform.modifications[4].label is None
    proteoform = proforma.parse(
        "{Glycan:Hex}[iTRAQ4plex]-EM[Oxidation]EVNES[Phospho]PE"
        "K[iTRAQ4plex]-[Methyl]"
    )[0]
    assert proteoform.sequence == "EMEVNESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 6
    )
    assert proteoform.modifications[0].mass == 162.052823418
    assert proteoform.modifications[0].position == "labile"
    assert (
        proteoform.modifications[0].source[0].composition[0].monosaccharide
        == "Hex"
    )
    assert proteoform.modifications[0].source[0].composition[0].count == 1
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 144.102063
    assert proteoform.modifications[1].position == "N-term"
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:214"
    assert proteoform.modifications[1].source[0].name == "iTRAQ4plex"
    assert proteoform.modifications[1].label is None
    assert proteoform.modifications[2].mass == 15.994915
    assert proteoform.modifications[2].position == 1
    assert (
        proteoform.modifications[2].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[2].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[2].source[0].name == "Oxidation"
    assert proteoform.modifications[2].label is None
    assert proteoform.modifications[3].mass == 79.966331
    assert proteoform.modifications[3].position == 6
    assert (
        proteoform.modifications[3].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[3].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[3].source[0].name == "Phospho"
    assert proteoform.modifications[3].label is None
    assert proteoform.modifications[4].mass == 144.102063
    assert proteoform.modifications[4].position == 9
    assert (
        proteoform.modifications[4].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[4].source[0].accession == "UNIMOD:214"
    assert proteoform.modifications[4].source[0].name == "iTRAQ4plex"
    assert proteoform.modifications[4].label is None
    assert proteoform.modifications[5].mass == 14.01565
    assert proteoform.modifications[5].position == "C-term"
    assert (
        proteoform.modifications[5].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[5].source[0].accession == "UNIMOD:34"
    assert proteoform.modifications[5].source[0].name == "Methyl"
    assert proteoform.modifications[5].label is None
    proteoform = proforma.parse("{Glycan:Hex}{Glycan:NeuAc}EMEVNESPEK")[0]
    assert proteoform.sequence == "EMEVNESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 162.052823418
    assert proteoform.modifications[0].position == "labile"
    assert (
        proteoform.modifications[0].source[0].composition[0].monosaccharide
        == "Hex"
    )
    assert proteoform.modifications[0].source[0].composition[0].count == 1
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 291.095416506
    assert proteoform.modifications[1].position == "labile"
    assert (
        proteoform.modifications[1].source[0].composition[0].monosaccharide
        == "NeuAc"
    )
    assert proteoform.modifications[1].source[0].composition[0].count == 1
    assert proteoform.modifications[1].label is None
    # Example: N-terminal or first AA acetylation?
    proteoform = proforma.parse("[#g1]-K[Acetyl#g1]EMEVTSESPEK")[0]
    assert proteoform.sequence == "KEMEVTSESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass is None
    assert proteoform.modifications[0].position == "N-term"
    assert proteoform.modifications[0].source is None
    assert proteoform.modifications[0].label.type == proforma.LabelType.GENERAL
    assert proteoform.modifications[0].label.label == "g1"
    assert proteoform.modifications[1].mass == 42.010565
    assert proteoform.modifications[1].position == 0
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:1"
    assert proteoform.modifications[1].source[0].name == "Acetyl"
    assert proteoform.modifications[1].label.type == proforma.LabelType.GENERAL
    assert proteoform.modifications[1].label.label == "g1"


# noinspection DuplicatedCode
# noinspection PyUnresolvedReferences
def test_proforma_ambiguous_position():
    # Unknown modification position.
    proteoform = proforma.parse("[Phospho]?EM[Oxidation]EVTSESPEK")[0]
    assert proteoform.sequence == "EMEVTSESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 79.966331
    assert proteoform.modifications[0].position == "unknown"
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[0].source[0].name == "Phospho"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 15.994915
    assert proteoform.modifications[1].position == 1
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[1].source[0].name == "Oxidation"
    assert proteoform.modifications[1].label is None
    proteoform = proforma.parse(
        "[Phospho][Phospho]?[Acetyl]-EM[Oxidation]EVTSESPEK"
    )[0]
    assert proteoform.sequence == "EMEVTSESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 4
    )
    assert proteoform.modifications[0].mass == 79.966331
    assert proteoform.modifications[0].position == "unknown"
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[0].source[0].name == "Phospho"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.966331
    assert proteoform.modifications[1].position == "unknown"
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[1].source[0].name == "Phospho"
    assert proteoform.modifications[1].label is None
    assert proteoform.modifications[2].mass == 42.010565
    assert proteoform.modifications[2].position == "N-term"
    assert (
        proteoform.modifications[2].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[2].source[0].accession == "UNIMOD:1"
    assert proteoform.modifications[2].source[0].name == "Acetyl"
    assert proteoform.modifications[2].label is None
    assert proteoform.modifications[3].mass == 15.994915
    assert proteoform.modifications[3].position == 1
    assert (
        proteoform.modifications[3].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[3].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[3].source[0].name == "Oxidation"
    assert proteoform.modifications[3].label is None
    proteoform = proforma.parse(
        "[Phospho]^2[Methyl]?[Acetyl]-EM[Oxidation]EVTSESPEK"
    )[0]
    assert proteoform.sequence == "EMEVTSESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 5
    )
    assert proteoform.modifications[0].mass == 79.966331
    assert proteoform.modifications[0].position == "unknown"
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[0].source[0].name == "Phospho"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.966331
    assert proteoform.modifications[1].position == "unknown"
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[1].source[0].name == "Phospho"
    assert proteoform.modifications[1].label is None
    assert proteoform.modifications[2].mass == 14.01565
    assert proteoform.modifications[2].position == "unknown"
    assert (
        proteoform.modifications[2].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[2].source[0].accession == "UNIMOD:34"
    assert proteoform.modifications[2].source[0].name == "Methyl"
    assert proteoform.modifications[2].label is None
    assert proteoform.modifications[3].mass == 42.010565
    assert proteoform.modifications[3].position == "N-term"
    assert (
        proteoform.modifications[3].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[3].source[0].accession == "UNIMOD:1"
    assert proteoform.modifications[3].source[0].name == "Acetyl"
    assert proteoform.modifications[3].label is None
    assert proteoform.modifications[4].mass == 15.994915
    assert proteoform.modifications[4].position == 1
    assert (
        proteoform.modifications[4].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[4].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[4].source[0].name == "Oxidation"
    assert proteoform.modifications[4].label is None
    proteoform = proforma.parse("[Phospho]^2?[Acetyl]-EM[Oxidation]EVTSESPEK")[
        0
    ]
    assert proteoform.sequence == "EMEVTSESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 4
    )
    assert proteoform.modifications[0].mass == 79.966331
    assert proteoform.modifications[0].position == "unknown"
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[0].source[0].name == "Phospho"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.966331
    assert proteoform.modifications[1].position == "unknown"
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[1].source[0].name == "Phospho"
    assert proteoform.modifications[1].label is None
    assert proteoform.modifications[2].mass == 42.010565
    assert proteoform.modifications[2].position == "N-term"
    assert (
        proteoform.modifications[2].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[2].source[0].accession == "UNIMOD:1"
    assert proteoform.modifications[2].source[0].name == "Acetyl"
    assert proteoform.modifications[2].label is None
    assert proteoform.modifications[3].mass == 15.994915
    assert proteoform.modifications[3].position == 1
    assert (
        proteoform.modifications[3].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[3].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[3].source[0].name == "Oxidation"
    assert proteoform.modifications[3].label is None
    # Multiple possible modification positions.
    proteoform = proforma.parse(
        "EM[Oxidation]EVT[#g1]S[#g1]ES[Phospho#g1]PEK"
    )[0]
    assert proteoform.sequence == "EMEVTSESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 4
    )
    assert proteoform.modifications[0].mass == 15.994915
    assert proteoform.modifications[0].position == 1
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[0].source[0].name == "Oxidation"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass is None
    assert proteoform.modifications[1].position == 4
    assert proteoform.modifications[1].source is None
    assert proteoform.modifications[1].label.type == proforma.LabelType.GENERAL
    assert proteoform.modifications[1].label.label == "g1"
    assert proteoform.modifications[2].mass is None
    assert proteoform.modifications[2].position == 5
    assert proteoform.modifications[2].source is None
    assert proteoform.modifications[2].label.type == proforma.LabelType.GENERAL
    assert proteoform.modifications[2].label.label == "g1"
    assert proteoform.modifications[3].mass == 79.966331
    assert proteoform.modifications[3].position == 7
    assert (
        proteoform.modifications[3].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[3].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[3].source[0].name == "Phospho"
    assert proteoform.modifications[3].label.type == proforma.LabelType.GENERAL
    assert proteoform.modifications[3].label.label == "g1"
    with pytest.raises(ValueError):
        proforma.parse("EM[Oxidation]EVT[#g1]S[Phospho#g1]ES[Phospho#g1]PEK")
    # Ranges of modification positions.
    proteoform = proforma.parse("PROT(EOSFORMS)[+19.0523]ISK")[0]
    assert proteoform.sequence == "PROTEOSFORMSISK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 19.0523
    assert proteoform.modifications[0].position == (4, 11)
    assert proteoform.modifications[0].source[0].mass == 19.0523
    assert proteoform.modifications[0].source[0].controlled_vocabulary is None
    assert proteoform.modifications[0].label is None
    proteoform = proforma.parse(
        "PROT(EOC[Carbamidomethyl]FORMS)[+19.0523]ISK"
    )[0]
    assert proteoform.sequence == "PROTEOCFORMSISK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 19.0523
    assert proteoform.modifications[0].position == (4, 11)
    assert proteoform.modifications[0].source[0].mass == 19.0523
    assert proteoform.modifications[0].source[0].controlled_vocabulary is None
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 57.021464
    assert proteoform.modifications[1].position == 6
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:4"
    assert proteoform.modifications[1].source[0].name == "Carbamidomethyl"
    assert proteoform.modifications[1].label is None
    proteoform = proforma.parse("P(ROT(EOSFORMS)[+19.0523]IS)[+19.0523]K")[0]
    assert proteoform.sequence == "PROTEOSFORMSISK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 19.0523
    assert proteoform.modifications[0].position == (1, 13)
    assert proteoform.modifications[0].source[0].mass == 19.0523
    assert proteoform.modifications[0].source[0].controlled_vocabulary is None
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 19.0523
    assert proteoform.modifications[1].position == (4, 11)
    assert proteoform.modifications[1].source[0].mass == 19.0523
    assert proteoform.modifications[1].source[0].controlled_vocabulary is None
    assert proteoform.modifications[1].label is None
    # Modification localization score.
    proteoform = proforma.parse(
        "EM[Oxidation]EVT[#g1(0.01)]S[#g1(0.09)]ES[Phospho#g1(0.90)]PEK"
    )[0]
    assert proteoform.sequence == "EMEVTSESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 4
    )
    assert proteoform.modifications[0].mass == 15.994915
    assert proteoform.modifications[0].position == 1
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[0].source[0].name == "Oxidation"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass is None
    assert proteoform.modifications[1].position == 4
    assert proteoform.modifications[1].source is None
    assert proteoform.modifications[1].label.type == proforma.LabelType.GENERAL
    assert proteoform.modifications[1].label.label == "g1"
    assert proteoform.modifications[1].label.score == 0.01
    assert proteoform.modifications[2].mass is None
    assert proteoform.modifications[2].position == 5
    assert proteoform.modifications[2].source is None
    assert proteoform.modifications[2].label.type == proforma.LabelType.GENERAL
    assert proteoform.modifications[2].label.label == "g1"
    assert proteoform.modifications[2].label.score == 0.09
    assert proteoform.modifications[3].mass == 79.966331
    assert proteoform.modifications[3].position == 7
    assert (
        proteoform.modifications[3].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[3].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[3].source[0].name == "Phospho"
    assert proteoform.modifications[3].label.type == proforma.LabelType.GENERAL
    assert proteoform.modifications[3].label.label == "g1"
    assert proteoform.modifications[3].label.score == 0.90
    proteoform = proforma.parse(
        "[Phospho#s1]?EM[Oxidation]EVT[#s1(0.01)]S[#s1(0.09)]ES[#s1(0.90)]PEK"
    )[0]
    assert proteoform.sequence == "EMEVTSESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 5
    )
    assert proteoform.modifications[0].mass == 79.966331
    assert proteoform.modifications[0].position == "unknown"
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[0].source[0].name == "Phospho"
    assert proteoform.modifications[0].label.type == proforma.LabelType.GENERAL
    assert proteoform.modifications[0].label.label == "s1"
    assert proteoform.modifications[1].mass == 15.994915
    assert proteoform.modifications[1].position == 1
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[1].source[0].name == "Oxidation"
    assert proteoform.modifications[1].label is None
    assert proteoform.modifications[2].mass is None
    assert proteoform.modifications[2].position == 4
    assert proteoform.modifications[2].source is None
    assert proteoform.modifications[2].label.type == proforma.LabelType.GENERAL
    assert proteoform.modifications[2].label.label == "s1"
    assert proteoform.modifications[2].label.score == 0.01
    assert proteoform.modifications[3].mass is None
    assert proteoform.modifications[3].position == 5
    assert proteoform.modifications[3].source is None
    assert proteoform.modifications[3].label.type == proforma.LabelType.GENERAL
    assert proteoform.modifications[3].label.label == "s1"
    assert proteoform.modifications[3].label.score == 0.09
    assert proteoform.modifications[4].mass is None
    assert proteoform.modifications[4].position == 7
    assert proteoform.modifications[4].source is None
    assert proteoform.modifications[4].label.type == proforma.LabelType.GENERAL
    assert proteoform.modifications[4].label.label == "s1"
    assert proteoform.modifications[4].label.score == 0.90
    # Scoring ranges of positions.
    proteoform = proforma.parse(
        "PROT(EOSFORMS)[+19.0523#g1(0.01)]ISK[#g1(0.99)]"
    )[0]
    assert proteoform.sequence == "PROTEOSFORMSISK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 19.0523
    assert proteoform.modifications[0].position == (4, 11)
    assert proteoform.modifications[0].source[0].mass == 19.0523
    assert proteoform.modifications[0].source[0].controlled_vocabulary is None
    assert proteoform.modifications[0].label.type == proforma.LabelType.GENERAL
    assert proteoform.modifications[0].label.label == "g1"
    assert proteoform.modifications[0].label.score == 0.01
    assert proteoform.modifications[1].mass is None
    assert proteoform.modifications[1].position == 14
    assert proteoform.modifications[1].source is None
    assert proteoform.modifications[1].label.type == proforma.LabelType.GENERAL
    assert proteoform.modifications[1].label.label == "g1"
    assert proteoform.modifications[1].label.score == 0.99
    proteoform = proforma.parse(
        "PR[#g1(0.91)]OT(EOC[Carbamidomethyl]FORMS)[+19.05233#g1(0.09)]ISK"
    )[0]
    assert proteoform.sequence == "PROTEOCFORMSISK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 3
    )
    assert proteoform.modifications[0].mass is None
    assert proteoform.modifications[0].position == 1
    assert proteoform.modifications[0].source is None
    assert proteoform.modifications[0].label.type == proforma.LabelType.GENERAL
    assert proteoform.modifications[0].label.label == "g1"
    assert proteoform.modifications[0].label.score == 0.91
    assert proteoform.modifications[1].mass == 19.05233
    assert proteoform.modifications[1].position == (4, 11)
    assert proteoform.modifications[1].source[0].mass == 19.05233
    assert proteoform.modifications[1].source[0].controlled_vocabulary is None
    assert proteoform.modifications[1].label.type == proforma.LabelType.GENERAL
    assert proteoform.modifications[1].label.label == "g1"
    assert proteoform.modifications[1].label.score == 0.09
    assert proteoform.modifications[2].mass == 57.021464
    assert proteoform.modifications[2].position == 6
    assert (
        proteoform.modifications[2].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[2].source[0].accession == "UNIMOD:4"
    assert proteoform.modifications[2].source[0].name == "Carbamidomethyl"
    assert proteoform.modifications[2].label is None
    proteoform = proforma.parse(
        "MPGLVDSNPAPPESQEKKPLK(PCCACPETKKARDACIIEKGEEHCGHLIEAHKECMRALGFKI)"
        "[Oxidation][Oxidation][half cystine][half cystine]"
    )[0]
    assert (
        proteoform.sequence
        == "MPGLVDSNPAPPESQEKKPLKPCCACPETKKARDACIIEKGEEHCGHLIEAHKECMRALGFKI"
    )
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 4
    )
    assert proteoform.modifications[0].mass == 15.994915
    assert proteoform.modifications[0].position == (21, 62)
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[0].source[0].name == "Oxidation"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 15.994915
    assert proteoform.modifications[1].position == (21, 62)
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[1].source[0].name == "Oxidation"
    assert proteoform.modifications[1].label is None
    assert proteoform.modifications[2].mass == -1.007825
    assert proteoform.modifications[2].position == (21, 62)
    assert proteoform.modifications[2].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[2].source[0].accession == "MOD:00798"
    assert proteoform.modifications[2].source[0].name == "half cystine"
    assert proteoform.modifications[2].label is None
    assert proteoform.modifications[3].mass == -1.007825
    assert proteoform.modifications[3].position == (21, 62)
    assert proteoform.modifications[3].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[3].source[0].accession == "MOD:00798"
    assert proteoform.modifications[3].source[0].name == "half cystine"
    assert proteoform.modifications[3].label is None


# noinspection DuplicatedCode
# noinspection PyUnresolvedReferences
def test_proforma_global_modification():
    proteoform = proforma.parse("<13C>ATPEILTVNSIGQLK")[0]
    with pytest.raises(NotImplementedError):
        print(proteoform.modifications[0].mass)
    proteoform = proforma.parse("<15N>ATPEILTVNSIGQLK")[0]
    with pytest.raises(NotImplementedError):
        print(proteoform.modifications[0].mass)
    proteoform = proforma.parse("<D>ATPEILTVNSIGQLK")[0]
    with pytest.raises(NotImplementedError):
        print(proteoform.modifications[0].mass)
    proteoform = proforma.parse("<13C><15N>ATPEILTVNSIGQLK")[0]
    with pytest.raises(NotImplementedError):
        print(proteoform.modifications[0].mass)
    proteoform = proforma.parse(
        "<[S-carboxamidomethyl-L-cysteine]@C>ATPEILTCNSIGCLK"
    )[0]
    assert proteoform.sequence == "ATPEILTCNSIGCLK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 57.021464
    assert proteoform.modifications[0].position == 7
    assert proteoform.modifications[0].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[0].source[0].accession == "MOD:01060"
    assert (
        proteoform.modifications[0].source[0].name
        == "S-carboxamidomethyl-L-cysteine"
    )
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 57.021464
    assert proteoform.modifications[1].position == 12
    assert proteoform.modifications[1].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[1].source[0].accession == "MOD:01060"
    assert (
        proteoform.modifications[1].source[0].name
        == "S-carboxamidomethyl-L-cysteine"
    )
    assert proteoform.modifications[1].label is None
    proteoform = proforma.parse("<[MOD:01090]@C>ATPEILTCNSIGCLK")[0]
    assert proteoform.sequence == "ATPEILTCNSIGCLK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 57.021464
    assert proteoform.modifications[0].position == 7
    assert proteoform.modifications[0].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[0].source[0].accession == "MOD:01090"
    assert (
        proteoform.modifications[0].source[0].name
        == "iodoacetamide derivatized amino-terminal residue"
    )
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 57.021464
    assert proteoform.modifications[1].position == 12
    assert proteoform.modifications[1].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[1].source[0].accession == "MOD:01090"
    assert (
        proteoform.modifications[1].source[0].name
        == "iodoacetamide derivatized amino-terminal residue"
    )
    assert proteoform.modifications[1].label is None
    proteoform = proforma.parse("<[Oxidation]@C,M>MTPEILTCNSIGCLK")[0]
    assert proteoform.sequence == "MTPEILTCNSIGCLK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 3
    )
    assert proteoform.modifications[0].mass == 15.994915
    assert proteoform.modifications[0].position == 0
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[0].source[0].name == "Oxidation"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 15.994915
    assert proteoform.modifications[1].position == 7
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[1].source[0].name == "Oxidation"
    assert proteoform.modifications[1].label is None
    assert proteoform.modifications[2].mass == 15.994915
    assert proteoform.modifications[2].position == 12
    assert (
        proteoform.modifications[2].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[2].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[2].source[0].name == "Oxidation"
    assert proteoform.modifications[2].label is None
    proteoform = proforma.parse(
        "<[MOD:01090]@C>[Phospho]?EM[Oxidation]EVTSECSPEK"
    )[0]
    assert proteoform.sequence == "EMEVTSECSPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 3
    )
    assert proteoform.modifications[0].mass == 79.966331
    assert proteoform.modifications[0].position == "unknown"
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[0].source[0].name == "Phospho"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 15.994915
    assert proteoform.modifications[1].position == 1
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[1].source[0].name == "Oxidation"
    assert proteoform.modifications[1].label is None
    assert proteoform.modifications[2].mass == 57.021464
    assert proteoform.modifications[2].position == 7
    assert proteoform.modifications[2].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[2].source[0].accession == "MOD:01090"
    assert (
        proteoform.modifications[2].source[0].name
        == "iodoacetamide derivatized amino-terminal residue"
    )
    assert proteoform.modifications[2].label is None
    proteoform = proforma.parse(
        "<[MOD:01090]@C>[Acetyl]-EM[Oxidation]EVTSECSPEK"
    )[0]
    assert proteoform.sequence == "EMEVTSECSPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 3
    )
    assert proteoform.modifications[0].mass == 42.010565
    assert proteoform.modifications[0].position == "N-term"
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:1"
    assert proteoform.modifications[0].source[0].name == "Acetyl"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 15.994915
    assert proteoform.modifications[1].position == 1
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[1].source[0].name == "Oxidation"
    assert proteoform.modifications[1].label is None
    assert proteoform.modifications[2].mass == 57.021464
    assert proteoform.modifications[2].position == 7
    assert proteoform.modifications[2].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[2].source[0].accession == "MOD:01090"
    assert (
        proteoform.modifications[2].source[0].name
        == "iodoacetamide derivatized amino-terminal residue"
    )
    assert proteoform.modifications[2].label is None


# noinspection PyUnresolvedReferences
def test_proforma_ambiguous_sequence():
    # FIXME: Support for amino acid sequence ambiguity.
    with pytest.raises(lark.exceptions.UnexpectedCharacters):
        proforma.parse("(?DQ)NGTWEM[Oxidation]ESNENFEGYM[Oxidation]K")
    with pytest.raises(lark.exceptions.UnexpectedCharacters):
        proforma.parse("(?N)NGTWEM[Oxidation]ESNENFEGYM[Oxidation]K")


# noinspection PyUnresolvedReferences
def test_proforma_info():
    proteoform = proforma.parse("ELV[INFO:AnyString]IS")[0]
    assert proteoform.sequence == "ELVIS"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass is None
    assert proteoform.modifications[0].position == 2
    assert proteoform.modifications[0].source[0].message == "AnyString"
    assert proteoform.modifications[0].label is None
    proteoform = proforma.parse("ELV[info:AnyString]IS")[0]
    assert proteoform.sequence == "ELVIS"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass is None
    assert proteoform.modifications[0].position == 2
    assert proteoform.modifications[0].source[0].message == "AnyString"
    assert proteoform.modifications[0].label is None
    proteoform = proforma.parse("ELVIS[Phospho|INFO:newly discovered]K")[0]
    assert proteoform.sequence == "ELVISK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 79.966331
    assert proteoform.modifications[0].position == 4
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[0].source[0].name == "Phospho"
    assert proteoform.modifications[0].source[1].message == "newly discovered"
    assert proteoform.modifications[0].label is None
    proteoform = proforma.parse(
        "ELVIS[Phospho|INFO:newly discovered|INFO:really awesome]K"
    )[0]
    assert proteoform.sequence == "ELVISK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 79.966331
    assert proteoform.modifications[0].position == 4
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[0].source[0].name == "Phospho"
    assert proteoform.modifications[0].source[1].message == "newly discovered"
    assert proteoform.modifications[0].source[2].message == "really awesome"
    assert proteoform.modifications[0].label is None


# noinspection DuplicatedCode
# noinspection PyUnresolvedReferences
def test_proforma_pipe():
    proteoform = proforma.parse("ELVIS[U:Phospho|+79.966331]K")[0]
    assert proteoform.sequence == "ELVISK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 79.966331
    assert proteoform.modifications[0].position == 4
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[0].source[0].name == "Phospho"
    assert proteoform.modifications[0].source[1].mass == 79.966331
    assert proteoform.modifications[0].label is None
    proteoform = proforma.parse("ELVIS[U:Phospho|Obs:+79.978]K")[0]
    assert proteoform.sequence == "ELVISK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 79.966331
    assert proteoform.modifications[0].position == 4
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[0].source[0].name == "Phospho"
    assert proteoform.modifications[0].source[1].mass == 79.978
    assert proteoform.modifications[0].label is None
    proteoform = proforma.parse("ELVIS[Phospho|O-phospho-L-serine]K")[0]
    assert proteoform.sequence == "ELVISK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 79.966331
    assert proteoform.modifications[0].position == 4
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[0].source[0].name == "Phospho"
    assert proteoform.modifications[0].source[1].controlled_vocabulary == "MOD"
    assert proteoform.modifications[0].source[1].accession == "MOD:00046"
    assert proteoform.modifications[0].source[1].name == "O-phospho-L-serine"
    assert proteoform.modifications[0].label is None
    proteoform = proforma.parse("ELVIS[UNIMOD:21|MOD:00046]K")[0]
    assert proteoform.sequence == "ELVISK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 79.966331
    assert proteoform.modifications[0].position == 4
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[0].source[0].name == "Phospho"
    assert proteoform.modifications[0].source[1].controlled_vocabulary == "MOD"
    assert proteoform.modifications[0].source[1].accession == "MOD:00046"
    assert proteoform.modifications[0].source[1].name == "O-phospho-L-serine"
    assert proteoform.modifications[0].label is None
    proteoform = proforma.parse("ELVIS[UNIMOD:21|Phospho]K")[0]
    assert proteoform.sequence == "ELVISK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 79.966331
    assert proteoform.modifications[0].position == 4
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[0].source[0].name == "Phospho"
    assert (
        proteoform.modifications[0].source[1].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[1].accession == "UNIMOD:21"
    assert proteoform.modifications[0].source[1].name == "Phospho"
    assert proteoform.modifications[0].label is None
    proteoform = proforma.parse(
        "ELVIS[Phospho|O-phospho-L-serine|Obs:+79.966]K"
    )[0]
    assert proteoform.sequence == "ELVISK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 79.966331
    assert proteoform.modifications[0].position == 4
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[0].source[0].name == "Phospho"
    assert proteoform.modifications[0].source[1].controlled_vocabulary == "MOD"
    assert proteoform.modifications[0].source[1].accession == "MOD:00046"
    assert proteoform.modifications[0].source[1].name == "O-phospho-L-serine"
    assert proteoform.modifications[0].source[2].mass == 79.966
    assert proteoform.modifications[0].label is None
    proteoform = proforma.parse("ELVIS[Obs:+79.966|Phospho|Sulfo]K")[0]
    assert proteoform.sequence == "ELVISK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 79.966
    assert proteoform.modifications[0].position == 4
    assert proteoform.modifications[0].source[0].mass == 79.966
    assert (
        proteoform.modifications[0].source[1].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[1].accession == "UNIMOD:21"
    assert proteoform.modifications[0].source[1].name == "Phospho"
    assert (
        proteoform.modifications[0].source[2].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[2].accession == "UNIMOD:40"
    assert proteoform.modifications[0].source[2].name == "Sulfo"
    assert proteoform.modifications[0].label is None


# noinspection DuplicatedCode
# noinspection PyUnresolvedReferences
def test_proforma_charge():
    proteoform = proforma.parse("EMEVEESPEK/2")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert proteoform.modifications is None
    assert proteoform.charge.charge == 2
    assert proteoform.charge.ions is None
    proteoform = proforma.parse("EM[U:Oxidation]EVEES[U:Phospho]PEK/3")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 15.994915
    assert proteoform.modifications[0].position == 1
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[0].source[0].name == "Oxidation"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.966331
    assert proteoform.modifications[1].position == 6
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[1].source[0].name == "Phospho"
    assert proteoform.modifications[1].label is None
    assert proteoform.charge.charge == 3
    assert proteoform.charge.ions is None
    proteoform = proforma.parse(
        "[U:iTRAQ4plex]-EM[U:Oxidation]EVNES[U:Phospho]PEK[U:iTRAQ4plex]"
        "-[U:Methyl]/3"
    )[0]
    assert proteoform.sequence == "EMEVNESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 5
    )
    assert proteoform.modifications[0].mass == 144.102063
    assert proteoform.modifications[0].position == "N-term"
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:214"
    assert proteoform.modifications[0].source[0].name == "iTRAQ4plex"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 15.994915
    assert proteoform.modifications[1].position == 1
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[1].source[0].name == "Oxidation"
    assert proteoform.modifications[1].label is None
    assert proteoform.modifications[2].mass == 79.966331
    assert proteoform.modifications[2].position == 6
    assert (
        proteoform.modifications[2].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[2].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[2].source[0].name == "Phospho"
    assert proteoform.modifications[2].label is None
    assert proteoform.modifications[3].mass == 144.102063
    assert proteoform.modifications[3].position == 9
    assert (
        proteoform.modifications[3].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[3].source[0].accession == "UNIMOD:214"
    assert proteoform.modifications[3].source[0].name == "iTRAQ4plex"
    assert proteoform.modifications[3].label is None
    assert proteoform.modifications[4].mass == 14.01565
    assert proteoform.modifications[4].position == "C-term"
    assert (
        proteoform.modifications[4].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[4].source[0].accession == "UNIMOD:34"
    assert proteoform.modifications[4].source[0].name == "Methyl"
    assert proteoform.modifications[4].label is None
    assert proteoform.charge.charge == 3
    assert proteoform.charge.ions is None
    proteoform = proforma.parse("EMEVEESPEK/2[+2Na+,+H+]")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert proteoform.modifications is None
    assert proteoform.charge.charge == 2
    assert (
        proteoform.charge.ions is not None and len(proteoform.charge.ions) == 2
    )
    assert proteoform.charge.ions[0].ion == "+2Na+"
    assert proteoform.charge.ions[1].ion == "+H+"
    proteoform = proforma.parse("EMEVEESPEK/1[+2Na+,-H+]")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert proteoform.modifications is None
    assert proteoform.charge.charge == 1
    assert (
        proteoform.charge.ions is not None and len(proteoform.charge.ions) == 2
    )
    assert proteoform.charge.ions[0].ion == "+2Na+"
    assert proteoform.charge.ions[1].ion == "-H+"
    proteoform = proforma.parse("EMEVEESPEK/-2[2I-]")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert proteoform.modifications is None
    assert proteoform.charge.charge == -2
    assert (
        proteoform.charge.ions is not None and len(proteoform.charge.ions) == 1
    )
    assert proteoform.charge.ions[0].ion == "2I-"
    proteoform = proforma.parse("EMEVEESPEK/-1[+e-]")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert proteoform.modifications is None
    assert proteoform.charge.charge == -1
    assert (
        proteoform.charge.ions is not None and len(proteoform.charge.ions) == 1
    )
    assert proteoform.charge.ions[0].ion == "+e-"


# noinspection PyUnresolvedReferences
def test_proforma_chimeric():
    proteoforms = proforma.parse("EMEVEESPEK/2+ELVISLIVER/3")
    proteoform = proteoforms[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert proteoform.modifications is None
    assert proteoform.charge.charge == 2
    assert proteoform.charge.ions is None
    proteoform = proteoforms[1]
    assert proteoform.sequence == "ELVISLIVER"
    assert proteoform.modifications is None
    assert proteoform.charge.charge == 3
    assert proteoform.charge.ions is None


def test_proforma_urlerror():
    # Connection error.
    with unittest.mock.patch("http.client.HTTPResponse.getcode") as mocked_req:
        mocked_req.return_value = 404
        cache_dir = proforma.cache_dir
        proforma.cache_dir = ".non_existing_cache"
        proteoform = proforma.parse("SEQUEN[Glycan:HexNAc1Hex2]CE")[0]
        with pytest.raises(URLError):
            print(proteoform.modifications[0].mass)
        shutil.rmtree(proforma.cache_dir, ignore_errors=True)
        proforma.cache_dir = cache_dir


# noinspection DuplicatedCode
# noinspection PyUnresolvedReferences
def test_proforma_cache():
    cache_dir = ".cache_test"
    cache_dir_original = proforma.cache_dir
    shutil.rmtree(cache_dir, ignore_errors=True)
    # Disable cache.
    proforma.cache_dir = None
    assert not os.path.isfile(os.path.join(cache_dir, "UNIMOD.pkl"))
    proteoform = proforma.parse("EM[U:Oxidation]EVEES[U:Phospho]PEK")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 15.994915
    assert proteoform.modifications[0].position == 1
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[0].source[0].name == "Oxidation"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.966331
    assert proteoform.modifications[1].position == 6
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[1].source[0].name == "Phospho"
    assert proteoform.modifications[1].label is None
    assert not os.path.isfile(os.path.join(cache_dir, "UNIMOD.pkl"))
    # All controlled vocabularies (and monosaccharides) from scratch
    # (without cache).
    # UNIMOD
    proteoform = proforma.parse("EM[U:Oxidation]EVEES[U:Phospho]PEK")[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 15.994915
    assert proteoform.modifications[0].position == 1
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[0].source[0].accession == "UNIMOD:35"
    assert proteoform.modifications[0].source[0].name == "Oxidation"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.966331
    assert proteoform.modifications[1].position == 6
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "UNIMOD"
    )
    assert proteoform.modifications[1].source[0].accession == "UNIMOD:21"
    assert proteoform.modifications[1].source[0].name == "Phospho"
    assert proteoform.modifications[1].label is None
    # MOD
    proteoform = proforma.parse(
        "EM[M:L-methionine sulfoxide]EVEES[M:O-phospho-L-serine]PEK"
    )[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 15.994915
    assert proteoform.modifications[0].position == 1
    assert proteoform.modifications[0].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[0].source[0].accession == "MOD:00719"
    assert (
        proteoform.modifications[0].source[0].name == "L-methionine sulfoxide"
    )
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.966331
    assert proteoform.modifications[1].position == 6
    assert proteoform.modifications[1].source[0].controlled_vocabulary == "MOD"
    assert proteoform.modifications[1].source[0].accession == "MOD:00046"
    assert proteoform.modifications[1].source[0].name == "O-phospho-L-serine"
    assert proteoform.modifications[1].label is None
    # RESID
    proteoform = proforma.parse(
        "EM[R:L-methionine sulfone]EVEES[R:O-phospho-L-serine]PEK"
    )[0]
    assert proteoform.sequence == "EMEVEESPEK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 2
    )
    assert proteoform.modifications[0].mass == 31.989829
    assert proteoform.modifications[0].position == 1
    assert (
        proteoform.modifications[0].source[0].controlled_vocabulary == "RESID"
    )
    assert proteoform.modifications[0].source[0].accession == "RESID:AA0251"
    assert proteoform.modifications[0].source[0].name == "L-methionine sulfone"
    assert proteoform.modifications[0].label is None
    assert proteoform.modifications[1].mass == 79.966331
    assert proteoform.modifications[1].position == 6
    assert (
        proteoform.modifications[1].source[0].controlled_vocabulary == "RESID"
    )
    assert proteoform.modifications[1].source[0].accession == "RESID:AA0037"
    assert proteoform.modifications[1].source[0].name == "O-phospho-L-serine"
    assert proteoform.modifications[1].label is None
    # XLMOD
    try:
        proteoform = proforma.parse("EMEVTK[X:DSS#XL1]SESPEK")[0]
        assert proteoform.sequence == "EMEVTKSESPEK"
        assert (
            proteoform.modifications is not None
            and len(proteoform.modifications) == 1
        )
        assert proteoform.modifications[0].mass == 138.06807961
        assert proteoform.modifications[0].position == 5
        assert (
            proteoform.modifications[0].source[0].controlled_vocabulary
            == "XLMOD"
        )
        assert proteoform.modifications[0].source[0].accession == "XLMOD:02001"
        assert proteoform.modifications[0].source[0].name == "DSS"
        assert proteoform.modifications[0].label.type == proforma.LabelType.XL
        assert proteoform.modifications[0].label.label == "XL1"
    except SyntaxError as e:
        # Only skip if this is the known XLMOD parsing issue
        if "XLMOD" in str(e) and "OBO file" in str(e):
            pytest.skip(
                f"XLMOD parsing failed due to known upstream OBO format issue: {e}"
            )
        else:
            # Re-raise any other SyntaxError as it's an actual bug
            raise
    # GNO
    proteoform = proforma.parse("NEEYN[G:G59626AS]K")[0]
    assert proteoform.sequence == "NEEYNK"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 1931.69
    assert proteoform.modifications[0].position == 4
    assert proteoform.modifications[0].source[0].controlled_vocabulary == "GNO"
    assert proteoform.modifications[0].source[0].accession == "GNO:G59626AS"
    assert proteoform.modifications[0].source[0].name == "G59626AS"
    assert proteoform.modifications[0].label is None
    # Monosaccharides
    proteoform = proforma.parse("SEQUEN[Glycan:HexNAc1Hex2]CE")[0]
    assert proteoform.sequence == "SEQUENCE"
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications) == 1
    )
    assert proteoform.modifications[0].mass == 527.185019356
    assert proteoform.modifications[0].position == 5
    assert (
        proteoform.modifications is not None
        and len(proteoform.modifications[0].source[0].composition) == 2
    )
    assert (
        proteoform.modifications[0].source[0].composition[0].monosaccharide
        == "HexNAc"
    )
    assert proteoform.modifications[0].source[0].composition[0].count == 1
    assert (
        proteoform.modifications[0].source[0].composition[1].monosaccharide
        == "Hex"
    )
    assert proteoform.modifications[0].source[0].composition[1].count == 2
    assert proteoform.modifications[0].label is None
    # Enable cache.
    proforma.cache_dir = cache_dir
    assert not os.path.isfile(os.path.join(cache_dir, "UNIMOD.pkl"))
    proteoform = proforma.parse("EM[U:Oxidation]EVEES[U:Phospho]PEK")[0]
    assert proteoform.modifications[0].mass == 15.994915
    assert os.path.isfile(os.path.join(cache_dir, "UNIMOD.pkl"))
    # Clear cache.
    proforma.clear_cache()
    assert not os.path.isfile(os.path.join(cache_dir, "UNIMOD.pkl"))
    proteoform = proforma.parse("EM[U:Oxidation]EVEES[U:Phospho]PEK")[0]
    assert proteoform.modifications[0].mass == 15.994915
    assert os.path.isfile(os.path.join(cache_dir, "UNIMOD.pkl"))
    # Clean-up.
    shutil.rmtree(cache_dir, ignore_errors=True)
    proforma.cache_dir = cache_dir_original


def test_proforma_import_cv():
    # Existing CVs.
    for cv_id in ("UNIMOD", "MOD", "RESID", "XLMOD", "GNO"):
        try:
            cv_from_id, cv_from_name = proforma._import_cv(
                cv_id, proforma.cache_dir
            )
            assert len(cv_from_id) > 0
            assert len(cv_from_name) > 0
        except SyntaxError as e:
            # Only skip XLMOD if it's the known OBO parsing issue
            if cv_id == "XLMOD" and "OBO file" in str(e):
                pytest.skip(
                    f"{cv_id} parsing failed due to known upstream OBO format issue: {e}"
                )
            else:
                # Re-raise for other vocabularies or unexpected errors
                raise
    mono = proforma._import_cv("mono", proforma.cache_dir)
    assert len(mono) > 0
    # Non-existing CV.
    with pytest.raises(ValueError):
        proforma._import_cv("THUS_IS_NO_CV", proforma.cache_dir)
    # Connection error.
    with unittest.mock.patch("http.client.HTTPResponse.getcode") as mocked_req:
        mocked_req.return_value = 404
        with pytest.raises(URLError):
            proforma._import_cv("UNIMOD", None)
