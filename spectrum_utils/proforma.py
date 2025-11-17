import abc
import collections
import copy
import datetime
import enum
import functools
import json
import math
import os
import pickle
import re
import urllib.request
from dataclasses import dataclass, field
from typing import (
    Any,
    BinaryIO,
    Dict,
    List,
    Optional,
    Set,
    Sequence,
    Tuple,
    Union,
)
from urllib.error import URLError

import platformdirs
import fastobo
import lark

try:
    import pyteomics.cmass as pmass
except ImportError:
    import pyteomics.mass as pmass


UNMODIFIED_PEPTIDE_REGEX = re.compile(r"^([A-Za-z]+)(/-?[0-9]+)?$")

# Set to None to disable caching.
cache_dir = platformdirs.user_cache_dir("spectrum_utils", False)


# noinspection PyArgumentList
LookupType = enum.Enum("LookupType", "ACCESSION NAME MASS")
# noinspection PyArgumentList
LabelType = enum.Enum("LabelType", "XL BRANCH GENERAL")


@dataclass
class Ion:
    ion: str


@dataclass
class Charge:
    charge: int
    ions: Optional[List[Ion]] = None


class ModificationSource(abc.ABC):
    @abc.abstractmethod
    def get_mass(self) -> float:
        """
        Get the mass of the modification from its source.

        Returns
        -------
        float
            The modification mass.
        """
        return math.nan


class CvEntry(ModificationSource):
    controlled_vocabulary: str
    accession: str
    name: str

    def __init__(
        self,
        controlled_vocabulary: Optional[str] = None,
        accession: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        self._controlled_vocabulary = controlled_vocabulary
        self._accession = accession
        self._name = name
        self._mass = None

    def __repr__(self):
        return (
            f"CvEntry(controlled_vocabulary={self.controlled_vocabulary}, "
            f"accession={self.accession}, name={self.name})"
        )

    @property
    def controlled_vocabulary(self) -> str:
        if self._controlled_vocabulary is None:
            for cv in ("UNIMOD", "MOD"):
                try:
                    self._read_from_cv(cv, name=self._name)
                    self.controlled_vocabulary = cv
                    break
                except KeyError:
                    pass
            else:
                raise KeyError(
                    f'Term "{self._name}" not found in UNIMOD or PSI-MOD'
                )
        return self._controlled_vocabulary

    @controlled_vocabulary.setter
    def controlled_vocabulary(self, controlled_vocabulary: str):
        self._controlled_vocabulary = controlled_vocabulary

    @property
    def accession(self) -> str:
        if self._accession is None:
            self._read_from_cv(self.controlled_vocabulary, name=self._name)
        return self._accession

    @accession.setter
    def accession(self, accession: str):
        self._accession = accession

    @property
    def name(self) -> str:
        if self._name is None:
            self._read_from_cv(
                self.controlled_vocabulary, accession=self._accession
            )
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    def get_mass(self) -> float:
        """
        Get the mass of the modification, as defined by its CV term.

        Returns
        -------
        float
            The modification mass.
        """
        if self._mass is None:
            # Force resolving from the CV.
            if self._accession is not None:
                self._read_from_cv(
                    self.controlled_vocabulary, accession=self._accession
                )
            elif self._name is not None:
                self._read_from_cv(self.controlled_vocabulary, name=self._name)
        return self._mass

    def _read_from_cv(
        self,
        cv: str,
        accession: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Read the term in its controlled vocabulary entry.

        Parameters
        ----------
        cv : str
            The controlled vocabulary identifier.
        accession : Optional[str]
            The accession to query the CV. Set as `None` to query by name,
        name : Optional[str]
            The name to query the CV. Set as `None` to query by accession,

        Raises
        ------
        KeyError
            If the term was not found in its controlled vocabulary.
        URLError
            If the controlled vocabulary could not be retrieved from its online
            resource.
        ValueError
            - If an unknown controlled vocabulary identifier is specified.
            - If no mass was specified for a GNO term or its parent terms.
        """
        cv_by_accession, cv_by_name = _import_cv(cv, cache_dir)
        key, lookup_type = None, None
        try:
            if accession is not None:
                key, lookup_type = self.accession, "accession"
                self._mass, self.name = cv_by_accession[key]
            elif name is not None:
                key, lookup_type = self.name, "name"
                self._mass, self.accession = cv_by_name[key]
            else:
                raise ValueError("No name or accession given")
        except KeyError:
            raise KeyError(
                f"Term {key} not found in the {cv} controlled vocabulary by "
                f"{lookup_type}"
            )


@dataclass
class Mass(ModificationSource):
    mass: float
    controlled_vocabulary: Optional[str] = None

    def get_mass(self) -> float:
        """
        Get the mass of the modification.

        Returns
        -------
        float
            The modification mass.
        """
        return self.mass


@dataclass
class Formula(ModificationSource):
    formula: Optional[str] = None
    isotopes: Optional[List[str]] = None
    _mass: float = field(default=None, init=False, repr=False)

    def get_mass(self) -> float:
        """
        Get the mass of the modification, computed from its molecular formula.

        Returns
        -------
        float
            The modification mass.

        Raises
        ------
        NotImplementedError
            Mass calculation using a molecular formula that contains isotopes
            (not supported by Pyteomics mass calculation).
        """
        if self._mass is None:
            if self.isotopes is not None:
                # FIXME: Add isotope support to Pyteomics.
                raise NotImplementedError(
                    "Mass calculation of molecular formulas with isotopes is "
                    "currently not supported"
                )
            # Make sure there are no spaces in the molecular formula (Pyteomics
            # mass calculation can't handle those).
            self._mass = pmass.calculate_mass(
                formula=self.formula.replace(" ", "")
            )
        return self._mass


@dataclass
class Monosaccharide:
    monosaccharide: str
    count: int


@dataclass
class Glycan(ModificationSource):
    composition: List[Monosaccharide]
    _mass: float = field(default=None, init=False, repr=False)

    def get_mass(self) -> float:
        """
        Get the mass of the modification, computed from its monosaccharide
        composition.

        Returns
        -------
        float
            The modification mass.

        Raises
        ------
        URLError
            If the monosaccharide definitions could not be retrieved from its
            online resource.
        """
        if self._mass is None:
            mono = _import_cv("mono", cache_dir)
            self._mass = sum(
                [mono[m.monosaccharide] * m.count for m in self.composition]
            )
        return self._mass


@dataclass
class Info:
    message: str


@dataclass
class Label:
    type: LabelType
    label: str
    score: Optional[float] = None


class Modification:
    mass: Optional[float]
    position: Union[int, Tuple[int, int], str]
    source: List[ModificationSource] = None
    label: Optional[Label] = None
    _mass: field(default=None, init=False, repr=False)

    def __init__(
        self,
        mass: Optional[float] = None,
        position: Optional[Union[int, Tuple[int, int], str]] = None,
        source: Optional[List[ModificationSource]] = None,
        label: Optional[Label] = None,
    ):
        self.position = position
        self.source = source
        self.label = label
        self._mass = mass

    def __repr__(self):
        return (
            f"Modification(mass={self._mass}, position={self.position}, "
            f"source={self.source}, label={self.label})"
        )

    @functools.cached_property
    def mass(self) -> Optional[float]:
        if self._mass is None and self.source is not None:
            for source in self.source:
                if isinstance(source, ModificationSource):
                    source_mass = source.get_mass()
                    if source_mass is not None and not math.isnan(source_mass):
                        self._mass = source_mass
                        break
        return self._mass


@dataclass
class Proteoform:
    sequence: str
    modifications: Optional[List[Modification]] = None
    charge: Optional[Charge] = None


# noinspection PyMethodMayBeStatic, PyPep8Naming
class ProFormaTransformer(lark.Transformer):
    _sequence: List[str]
    _modifications: List[Modification]
    _global_modifications: Dict[str, List[Modification]]
    _range_pos: List[int]
    _preferred_labels = Set[str]

    def __init__(self):
        super().__init__()
        self._sequence, self._modifications = [], []
        self._global_modifications = collections.defaultdict(list)
        self._range_pos, self._preferred_labels = [], set()

    def proforma(self, tree) -> List[Proteoform]:
        return [
            proteoform
            for proteoform in tree
            if isinstance(proteoform, Proteoform)
        ]

    def proteoform(self, tree) -> Proteoform:
        sequence = "".join(self._sequence)
        # Apply global modifications to the relevant residues.
        for i, aa in enumerate(sequence):
            if aa in self._global_modifications:
                for mod in self._global_modifications[aa]:
                    mod = copy.copy(mod)
                    mod.position = i
                    self._modifications.append(mod)
        charge = tree[-1]
        self._modifications.sort(key=_modification_sort_key)
        proteoform = Proteoform(
            sequence=sequence,
            modifications=(
                self._modifications if len(self._modifications) > 0 else None
            ),
            charge=charge,
        )
        # Reset class variables.
        self._sequence, self._modifications = [], []
        self._global_modifications = collections.defaultdict(list)
        self._range_pos, self._preferred_labels = [], set()
        return proteoform

    def peptide(self, _) -> None:
        # Residues and modifications have been written to the class variables.
        pass

    def aa(self, tree) -> None:
        position = len(self._sequence)
        self._sequence.append(tree[0])
        # An amino acid token can be followed by (i) a modification on that
        # residue, or (ii) a label (linking it to another modified residue).
        if isinstance(tree[1], Label):
            # noinspection PyArgumentList
            self._modifications.append(
                Modification(position=position, label=tree[1])
            )
        elif isinstance(tree[1], Modification):
            tree[1].position = position
            self._modifications.append(tree[1])

    def AA(self, token) -> str:
        return token.value.upper()

    def mod_global(self, mods) -> None:
        if len(mods) == 1:
            # Global isotope.
            # noinspection PyArgumentList
            self._modifications.append(
                Modification(
                    position="global", source=[Formula(isotopes=mods)]
                )
            )
        else:
            # Global modification on a specific residue.
            for aa in mods[1:]:
                self._global_modifications[aa].append(mods[0])

    def ISOTOPE(self, token) -> str:
        return token.value

    def mod_unknown_pos(self, mods) -> None:
        for mod in mods:
            if mod is None:
                continue
            elif isinstance(mod, Modification):
                mod.position = "unknown"
                self._modifications.append(mod)
            else:
                # Modification count.
                for _ in range(int(mod) - 1):
                    self._modifications.append(
                        copy.copy(self._modifications[-1])
                    )

    def mod(self, mod_annotations) -> Modification:
        # noinspection PyArgumentList
        mod = Modification(source=[])
        for mod_annotation in mod_annotations:
            if isinstance(mod_annotation, Label):
                if mod_annotation.label in self._preferred_labels:
                    raise ValueError(
                        "There should only be a single preferred location per "
                        "possible set of modification positions"
                    )
                else:
                    self._preferred_labels.add(mod_annotation.label)
                mod.label = mod_annotation
            else:
                mod.source.append(mod_annotation)
        return mod

    def mod_labile(self, mod_annotations) -> None:
        mod = self.mod(mod_annotations)
        mod.position = "labile"
        self._modifications.append(mod)

    def MOD_COUNT(self, token) -> int:
        return int(token.value)

    def mod_n_term(self, mods) -> None:
        for mod in mods:
            if isinstance(mod, Label):
                # noinspection PyArgumentList
                mod = Modification(label=mod)
            mod.position = "N-term"
            self._modifications.append(mod)

    def mod_c_term(self, mods) -> None:
        for mod in mods:
            if isinstance(mod, Label):
                # noinspection PyArgumentList
                mod = Modification(label=mod)
            mod.position = "C-term"
            self._modifications.append(mod)

    def mod_range(self, tree) -> None:
        _, position, *mods = tree
        for mod in mods:
            mod.position = position
            self._modifications.append(mod)

    def mod_range_pos(self, _) -> Tuple[int, int]:
        return self._range_pos.pop(), len(self._sequence) - 1

    def MOD_RANGE_L(self, _) -> None:
        self._range_pos.append(len(self._sequence))

    def mod_name(self, tree) -> CvEntry:
        cv, name = tree if len(tree) == 2 else (None, tree[0])
        # noinspection PyArgumentList
        return CvEntry(cv, name=name)

    def CV_ABBREV_OPT(self, token) -> str:
        return self.CV_ABBREV(token)

    def CV_ABBREV(self, token) -> str:
        return {
            "U": "UNIMOD",
            "M": "MOD",
            "R": "RESID",
            "X": "XLMOD",
            "G": "GNO",
        }[token.value.upper()]

    def mod_accession(self, tree) -> CvEntry:
        # noinspection PyArgumentList
        return CvEntry(tree[0], f"{tree[0]}:{tree[1]}")

    def CV_NAME(self, token) -> str:
        return token.value

    def mod_mass(self, tree) -> Mass:
        if len(tree) == 2:
            if tree[0] == "Obs":
                return Mass(mass=tree[1])
            else:
                return Mass(mass=tree[1], controlled_vocabulary=tree[0])
        else:
            return Mass(mass=tree[-1])

    def MOD_MASS_OBS(self, _) -> str:
        return "Obs"

    def MOD_MASS(self, token) -> float:
        return float(token.value)

    def mod_formula(self, tree) -> Formula:
        *isotopes, formula = tree
        return Formula(
            formula=formula, isotopes=isotopes if isotopes else None
        )

    def FORMULA(self, token) -> str:
        return token.value

    def mod_glycan(self, tree) -> Glycan:
        # Ignore optional whitespace in monosaccharide composition.
        return Glycan(
            composition=[t for t in tree if isinstance(t, Monosaccharide)]
        )

    def monosaccharide(self, tree) -> Monosaccharide:
        return Monosaccharide(
            tree[0].value, int(tree[1].value) if len(tree) > 1 else 1
        )

    def info(self, tree) -> Info:
        return Info(tree[0])

    def mod_label(self, tree) -> Label:
        return Label(
            type=tree[0][0],
            label=tree[0][1],
            score=tree[1] if len(tree) > 1 else None,
        )

    def MOD_LABEL_XL(self, token) -> Tuple[LabelType, str]:
        return LabelType.XL, token.value

    def MOD_LABEL_BRANCH(self, token) -> Tuple[LabelType, str]:
        return LabelType.BRANCH, token.value

    def MOD_LABEL(self, token) -> Tuple[LabelType, str]:
        return LabelType.GENERAL, token.value

    def MOD_SCORE(self, token) -> float:
        return float(token)

    def charge(self, tree) -> Charge:
        return Charge(
            charge=int(tree[0].value), ions=tree[1] if len(tree) > 1 else None
        )

    def ion(self, tree) -> List[Ion]:
        return [Ion(ion) for ion in tree if ion is not None]

    def TEXT(self, token) -> str:
        return token.value


def _modification_sort_key(mod: Modification):
    """
    Key to sort modifications.

    The sort order is:
    1. Unknown modifications.
    2. Labile modifications.
    3. C-terminal modifications.
    4. Modifications on specific amino acids and modification ranges based on
    the start of the range.
    5. N-terminal modifications.
    """
    if isinstance(mod.position, int):
        return mod.position
    elif isinstance(mod.position, tuple):
        return mod.position[0]
    elif mod.position == "N-term":
        return -1
    elif mod.position == "C-term":
        return math.inf
    elif mod.position == "labile":
        return -2
    elif mod.position == "unknown":
        return -3
    elif mod.position == "global":
        return -4


@functools.lru_cache(1)
def _build_parser() -> lark.Lark:
    """Build a lark parser for proforma sequences.

    This function also caches the parser in-memory, thus loading it only
    once per process.
    """
    dir_name = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_name, "proforma.ebnf")) as f_in:
        parser = lark.Lark(
            f_in.read(),
            start="proforma",
            parser="earley",
            lexer="dynamic_complete",
            import_paths=[dir_name],
        )
    return parser


def parse(proforma: str) -> List[Proteoform]:
    """
    Parse a ProForma-encoded string.

    The string is parsed by building an abstract syntax tree to interpret the
    ProForma language. Next, modifications are localized to residue positions
    and translated into Python objects.

    Parameters
    ----------
    proforma : str
        The ProForma string.

    Returns
    -------
    List[Proteoform]
        A list of proteoforms with their modifications (and additional)
        information specified by the ProForma string.

    Raises
    ------
    URLError
        If a controlled vocabulary could not be retrieved from its online
        resource.
    KeyError
        If a term was not found in its controlled vocabulary.
    NotImplementedError
        Mass calculation using a molecular formula that contains isotopes (not
        supported by Pyteomics mass calculation).
    ValueError
        If no mass was specified for a GNO term or its parent terms.
    """
    match_unmod = UNMODIFIED_PEPTIDE_REGEX.match(proforma)
    if match_unmod is not None:
        # Fast path for unmodified peptides.
        charge = match_unmod.group(2)
        if charge is not None:
            charge = Charge(int(charge[1:]))
        return [
            Proteoform(sequence=match_unmod.group(1).upper(), charge=charge)
        ]

    parser = _build_parser()
    # noinspection PyUnresolvedReferences
    try:
        parsed = parser.parse(proforma)
        parsed = ProFormaTransformer().transform(parsed)
        return parsed
    except lark.visitors.VisitError as e:
        raise e.orig_exc


@functools.lru_cache
def _import_cv(
    cv_id: str, cache: Optional[str]
) -> Union[
    Tuple[Dict[str, Tuple[float, str]], Dict[str, Tuple[float, str]]],
    Dict[str, float],
]:
    """
    Import a ProForma controlled vocabulary from its online resource.

    Parameters
    ----------
    cv_id : str
        The controlled vocabulary identifier.
    cache : Optional[str]
        Directory used to cache downloaded controlled vocabularies, or None to
        disable caching.

    Returns
    -------
    Union[
        Tuple[Dict[str, Tuple[float, str]], Dict[str, Tuple[float, str]]],
        Dict[str, float],
    ]
        - For modification controlled vocabularies:
          A tuple with mappings (i) from term accession to modification mass
          and term name, and (ii) from term name to modification mass and term
          accession.
        - For the monosaccharide controlled vocabulary:
          A dictionary with as keys the monosaccharide strings and as values
          the corresponding masses.

    Raises
    ------
    URLError
        If the controlled vocabulary could not be retrieved from its online
        resource.
    ValueError
        - If an unknown controlled vocabulary identifier is specified.
        - If no mass was specified for a GNO term or its parent terms.
    """
    if cv_id == "UNIMOD":
        url = "http://www.unimod.org/obo/unimod.obo"
    elif cv_id in ("MOD", "RESID"):
        # RESID is not available as a separate entity anymore but is part of
        # PSI-MOD.
        url = (
            "https://raw.githubusercontent.com/HUPO-PSI/psi-mod-CV/master/"
            "PSI-MOD.obo"
        )
    elif cv_id == "XLMOD":
        url = (
            "https://raw.githubusercontent.com/HUPO-PSI/xlmod-CV/main/"
            "XLMOD.obo"
        )
    elif cv_id == "GNO":
        url = (
            "https://github.com/glygen-glycan-data/GNOme/releases/latest/"
            "download/GNOme.obo"
        )
    elif cv_id == "mono":
        url = (
            "https://raw.githubusercontent.com/HUPO-PSI/ProForma/master/"
            "monosaccharides/mono.obo.json"
        )
    else:
        raise ValueError(f"Unknown controlled vocabulary: {cv_id}")

    # Try to read the CV from the cache first.
    cv_cached = _load_from_cache(cache, f"{cv_id}.pkl")
    if cv_cached is not None:
        cv, date_cache = cv_cached
        # Check that the cached CV is up to date.
        match = re.fullmatch(
            r"^https://(?:raw\.githubusercontent|github)\.com/"
            r"([^/]+)/([^/]+)/(?:master|releases/latest/download)/(.+)$",
            url,
        )
        if match is not None:
            owner_name, repo_name, filename = match.group(1, 2, 3)
            try:
                with urllib.request.urlopen(
                    f"https://api.github.com/repos/{owner_name}/{repo_name}/"
                    f"commits?path={filename}&per_page=1&page=1"
                ) as response:
                    if response.getcode() < 400:
                        date_url = datetime.datetime.strptime(
                            json.load(response)[0]["commit"]["author"]["date"],
                            "%Y-%m-%dT%H:%M:%SZ",
                        )
                        if date_cache >= date_url:
                            return cv
            except urllib.error.HTTPError:
                pass
        else:
            # Just use the cached CV if we can't compare timestamps.
            return cv

    # If we're here it means that we should retrieve the CV from its URL.
    with urllib.request.urlopen(url) as response:
        if 400 <= response.getcode() < 600:
            raise URLError(
                f"Failed to retrieve the {cv_id} controlled vocabulary from "
                f"its URL"
            )
        if cv_id in ("UNIMOD", "MOD", "RESID", "XLMOD", "GNO"):
            cv = _parse_obo(response, cv_id)
        elif cv_id == "mono":
            mono_json = json.loads(
                response.read().decode(
                    response.info().get_param("charset", "utf-8")
                )
            )
            cv = {}
            for term in mono_json["terms"].values():
                mass = float(term["has_monoisotopic_mass"])
                cv[term["name"]] = mass
                for synonym in term.get("synonym", []):
                    cv[synonym] = mass
    # Save to the cache if enabled.
    _store_in_cache(cache, f"{cv_id}.pkl", (cv, datetime.datetime.now(datetime.timezone.utc)))
    return cv


def _parse_obo(
    obo_fh: BinaryIO, cv_id: str
) -> Tuple[Dict[str, Tuple[float, str]], Dict[str, Tuple[float, str]]]:
    """
    Parse an OBO controlled vocabulary.

    Supported OBO files are: UNIMOD, MOD (including RESID), XLMOD, GNO.

    Parameters
    ----------
    obo_fh : BinaryIO
        The OBO file handle.
    cv_id : str
        The controlled vocabulary identifier.

    Returns
    -------
    Tuple[Dict[str, Tuple[float, str]], Dict[str, Tuple[float, str]]]
        A tuple with mappings (i) from term accession to modification mass and
        term name, and (ii) from term name to modification mass and term
        accession.

    Raises
    ------
    ValueError
        If no mass was specified for a GNO term or its parent terms.
    """
    cv_by_accession, cv_by_name, gno_graph = {}, {}, {}
    for frame in fastobo.load(obo_fh):
        term_accession, term_name, term_mass = str(frame.id), None, None
        if isinstance(frame, fastobo.term.TermFrame):
            for clause in frame:
                if isinstance(clause, fastobo.term.NameClause):
                    term_name = clause.name.strip()
                    if cv_id == "GNO" and "molecular weight" in term_name:
                        term_mass = float(
                            term_name[
                                term_name.rindex("weight ")
                                + 7 : term_name.rindex(" Da")
                            ]
                        )
                elif cv_id == "RESID" and isinstance(
                    clause, fastobo.term.DefClause
                ):
                    for xref in clause.xrefs:
                        if xref.id.prefix == "RESID":
                            term_accession = str(xref.id)
                            break
                elif isinstance(clause, fastobo.term.XrefClause):
                    term_xref = clause.raw_value()
                    if (
                        cv_id == "UNIMOD" and "delta_mono_mass" in term_xref
                    ) or (
                        cv_id in ("MOD", "RESID")
                        and "DiffMono" in term_xref
                        and not term_xref.endswith('"none"')
                    ):
                        term_mass = float(
                            term_xref[
                                term_xref.index('"') + 1 : term_xref.rindex(
                                    '"'
                                )
                            ]
                        )
                elif (
                    cv_id == "XLMOD"
                    and isinstance(clause, fastobo.term.PropertyValueClause)
                    and (
                        (hasattr(clause.property_value.relation, 'prefix') and 
                         clause.property_value.relation.prefix == "monoIsotopicMass") or
                        str(clause.property_value.relation) == "monoIsotopicMass"
                    )
                ):
                    term_mass = float(clause.property_value.value)
                elif cv_id == "GNO" and isinstance(
                    clause, fastobo.term.IsAClause
                ):
                    gno_graph[term_accession] = term_name, str(clause.term)
            if (
                term_accession.startswith(f"{cv_id}:")
                and term_mass is not None
            ):
                cv_by_accession[term_accession] = term_mass, term_name
                if term_name is not None:
                    cv_by_name[term_name] = term_mass, term_accession
    if cv_id == "GNO":
        for term_accession, (term_name, parent_accession) in gno_graph.items():
            if term_accession in cv_by_accession:
                continue
            terms = [(term_accession, term_name)]
            while (
                parent_accession not in cv_by_accession
                and parent_accession in gno_graph
            ):
                parent_name, grandparent_accession = gno_graph[
                    parent_accession
                ]
                terms.append((parent_accession, parent_name))
                parent_accession = grandparent_accession
            if parent_accession not in cv_by_accession:
                raise ValueError(
                    f"No mass found for term {term_accession} in the GNO "
                    f"controlled vocabulary"
                )
            term_mass = cv_by_accession[parent_accession][0]
            for add_accession, add_name in terms:
                cv_by_accession[add_accession] = term_mass, add_name
                cv_by_name[add_name] = term_mass, add_accession
    return cv_by_accession, cv_by_name


def _store_in_cache(cache: Optional[str], filename: str, obj: Any) -> None:
    """
    Store an object in the cache.

    Parameters
    ----------
    cache : Optional[str]
        Directory where the cached objects are stored, or None to disable
        caching.
    filename : str
        Filename of the stored cache object.
    obj : Any
        Object to store in the cache.
    """
    if cache is not None:
        os.makedirs(cache, exist_ok=True)
        with open(os.path.join(cache, filename), "wb") as f_out:
            pickle.dump(obj, f_out, pickle.HIGHEST_PROTOCOL)


def _load_from_cache(cache: Optional[str], filename: str) -> Optional[Any]:
    """
    Load an object from the cache.

    Parameters
    ----------
    cache : Optional[str]
        Directory where the cached objects are stored, or None to disable
        caching.
    filename : str
        Filename of the stored cache object.

    Returns
    -------
    Any
        The object retrieved from the cache, or None if not found.
    """
    if cache is not None:
        cache_filename = os.path.join(cache, filename)
        if os.path.isfile(cache_filename):
            with open(cache_filename, "rb") as f_in:
                return pickle.load(f_in)
    return None


def clear_cache(resource_ids: Optional[Sequence[str]] = None) -> None:
    """
    Clear the downloaded resources from the cache.

    Parameters
    ----------
    resource_ids : Optional[Sequence[str]]
        Identifiers of the resources to remove from the cache. If None, all
        known files will be removed.
    """
    # Clear the in-memory cache.
    _import_cv.cache_clear()
    # Clear the on-disk cache.
    if cache_dir is not None:
        if resource_ids is None:
            resource_ids = ("UNIMOD", "MOD", "RESID", "XLMOD", "GNO", "mono")
        for cv_id in resource_ids:
            filename = os.path.join(cache_dir, f"{cv_id}.pkl")
            if os.path.isfile(filename):
                os.remove(filename)
