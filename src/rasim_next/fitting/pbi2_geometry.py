"""Ideal shared-metric PbI2 polytype landmarks for geometry qualification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rasim_next.core.contracts import canonical_revision_sha256
from rasim_next.core.layer_order import CommensurateLayerOrder
from rasim_next.fitting.geometry import (
    ExactTagGeometryModel,
    LayerLMarkerDefinition,
    LayerLMarkerKey,
    LayerLMarkerPrediction,
)
from rasim_next.ordered.motifs import extract_pbi2_motifs
from rasim_next.stacking import Parent, RegistryPhaseModel

PBI2_IDEAL_PARENTS = (
    Parent.TWO_H,
    Parent.FOUR_H_PLUS,
    Parent.FOUR_H_MINUS,
    Parent.SIX_H_PLUS,
    Parent.SIX_H_MINUS,
)
_PBI2_LANDMARK_MODEL_ID = "pbi2.declared_single_trilayer_metric.landmarks.v1"

_PARENT_PERIOD = {
    Parent.TWO_H: 1,
    Parent.FOUR_H_PLUS: 2,
    Parent.FOUR_H_MINUS: 2,
    Parent.SIX_H_PLUS: 3,
    Parent.SIX_H_MINUS: 3,
}


@dataclass(frozen=True, slots=True)
class Pbi2ParentLandmarkContribution:
    """One ideal parent that can scatter at a signed-rod rational landmark."""

    parent: Parent
    signed_rod_hk: tuple[int, int]
    registry_sector: int
    conventional_cell_l: int

    def __post_init__(self) -> None:
        if self.parent not in PBI2_IDEAL_PARENTS:
            raise ValueError("unsupported ideal PbI2 parent")
        rod = tuple(self.signed_rod_hk)
        if len(rod) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in rod
        ):
            raise ValueError("signed_rod_hk must be one integer (h, k) pair")
        if self.registry_sector not in {-1, 0, 1}:
            raise ValueError("registry_sector must be -1, 0, or 1")
        if isinstance(self.conventional_cell_l, bool) or not isinstance(
            self.conventional_cell_l,
            (int, np.integer),
        ):
            raise TypeError("conventional_cell_l must be an integer")
        object.__setattr__(self, "signed_rod_hk", (int(rod[0]), int(rod[1])))
        object.__setattr__(self, "conventional_cell_l", int(self.conventional_cell_l))


def pbi2_ideal_parent_landmark_contributions(
    *,
    signed_rod_hk: tuple[int, int],
    layer_order: CommensurateLayerOrder,
    parents: tuple[Parent, ...] = PBI2_IDEAL_PARENTS,
) -> tuple[Pbi2ParentLandmarkContribution, ...]:
    """Return exact pure-parent support in the common one-trilayer coordinate."""

    rod = tuple(signed_rod_hk)
    if len(rod) != 2 or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in rod
    ):
        raise ValueError("signed_rod_hk must be one integer (h, k) pair")
    if not isinstance(layer_order, CommensurateLayerOrder):
        raise TypeError("layer_order must be CommensurateLayerOrder")
    selected = tuple(Parent(parent) for parent in parents)
    if len(set(selected)) != len(selected) or any(
        parent not in PBI2_IDEAL_PARENTS for parent in selected
    ):
        raise ValueError("parents must be unique supported ideal PbI2 parents")
    residue = (int(rod[0]) + 2 * int(rod[1])) % 3
    sector = -1 if residue == 2 else residue
    contributions: list[Pbi2ParentLandmarkContribution] = []
    for parent in selected:
        period = _PARENT_PERIOD[parent]
        scaled_numerator = period * layer_order.numerator
        if scaled_numerator % layer_order.denominator:
            continue
        cell_l = scaled_numerator // layer_order.denominator
        supported = False
        if parent is Parent.TWO_H:
            supported = True
        elif parent in {Parent.FOUR_H_PLUS, Parent.FOUR_H_MINUS}:
            supported = sector != 0 or cell_l % 2 == 0
        elif parent is Parent.SIX_H_PLUS:
            supported = (cell_l + sector) % 3 == 0
        elif parent is Parent.SIX_H_MINUS:
            supported = (cell_l - sector) % 3 == 0
        if supported:
            contributions.append(
                Pbi2ParentLandmarkContribution(
                    parent=parent,
                    signed_rod_hk=(int(rod[0]), int(rod[1])),
                    registry_sector=sector,
                    conventional_cell_l=cell_l,
                )
            )
    return tuple(contributions)


def _pbi2_landmark_catalogue_revision(
    *,
    prediction: LayerLMarkerPrediction,
    parent_contributions: tuple[tuple[Pbi2ParentLandmarkContribution, ...], ...],
    selected_parents: tuple[Parent, ...],
    source_cif_sha256: str,
    reciprocal_basis_revision: str,
    geometry_context_revision: str,
) -> str:
    parent_code = {parent: index for index, parent in enumerate(PBI2_IDEAL_PARENTS)}
    contributor_rows = np.asarray(
        [
            (landmark_index, h, k)
            for landmark_index, definition in enumerate(prediction.definitions)
            for h, k in definition.contributing_rod_hk
        ],
        dtype=np.int64,
    )
    parent_rows = np.asarray(
        [
            (
                landmark_index,
                parent_code[item.parent],
                item.signed_rod_hk[0],
                item.signed_rod_hk[1],
                item.registry_sector,
                item.conventional_cell_l,
            )
            for landmark_index, group in enumerate(parent_contributions)
            for item in group
        ],
        dtype=np.int64,
    )
    return canonical_revision_sha256(
        ("definition_id", _PBI2_LANDMARK_MODEL_ID),
        ("source_cif_sha256", source_cif_sha256),
        ("reciprocal_basis_revision", reciprocal_basis_revision),
        ("geometry_context_revision", geometry_context_revision),
        ("registry_phase_model", RegistryPhaseModel.FORWARD_H_PLUS_2K.value),
        ("parents", tuple(parent.value for parent in selected_parents)),
        (
            "layer_order_numerator",
            np.asarray([key.layer_order.numerator for key in prediction.keys], dtype=np.int64),
        ),
        (
            "layer_order_denominator",
            np.asarray([key.layer_order.denominator for key in prediction.keys], dtype=np.int64),
        ),
        ("family_m", np.asarray([key.family_m for key in prediction.keys], dtype=np.int64)),
        ("branch", np.asarray([key.branch for key in prediction.keys], dtype=np.int64)),
        ("root_sign", np.asarray([key.root_sign for key in prediction.keys], dtype=np.int64)),
        ("contributing_rod_rows", contributor_rows),
        ("parent_contribution_rows", parent_rows),
        ("coordinates_px", prediction.coordinates_px),
    )


@dataclass(frozen=True, slots=True)
class Pbi2PolytypeLandmarkCatalogue:
    """Geometry-visible ideal parent sites in one declared 2H layer metric."""

    prediction: LayerLMarkerPrediction
    parent_contributions: tuple[tuple[Pbi2ParentLandmarkContribution, ...], ...]
    selected_parents: tuple[Parent, ...]
    source_cif_sha256: str
    reciprocal_basis_revision: str
    geometry_context_revision: str
    catalogue_revision: str
    model_id: str = _PBI2_LANDMARK_MODEL_ID
    registry_phase_model: RegistryPhaseModel = RegistryPhaseModel.FORWARD_H_PLUS_2K

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, LayerLMarkerPrediction):
            raise TypeError("prediction must be LayerLMarkerPrediction")
        contributions = tuple(tuple(group) for group in self.parent_contributions)
        selected_parents = tuple(Parent(parent) for parent in self.selected_parents)
        if len(set(selected_parents)) != len(selected_parents) or any(
            parent not in PBI2_IDEAL_PARENTS for parent in selected_parents
        ):
            raise ValueError("selected_parents must be unique ideal PbI2 parents")
        if len(contributions) != len(self.prediction.definitions) or any(
            not group or any(not isinstance(item, Pbi2ParentLandmarkContribution) for item in group)
            for group in contributions
        ):
            raise ValueError("parent contributions must align with every catalogue landmark")
        for definition, group in zip(
            self.prediction.definitions,
            contributions,
            strict=True,
        ):
            if {item.signed_rod_hk for item in group} != set(definition.contributing_rod_hk):
                raise ValueError("physical landmark rods must equal the exact signed-rod support")
            expected = tuple(
                item
                for rod_hk in definition.contributing_rod_hk
                for item in pbi2_ideal_parent_landmark_contributions(
                    signed_rod_hk=rod_hk,
                    layer_order=definition.key.layer_order,
                    parents=selected_parents,
                )
            )
            if group != expected:
                raise ValueError("parent contributions do not match exact signed-rod support")
        for name in (
            "source_cif_sha256",
            "reciprocal_basis_revision",
            "geometry_context_revision",
            "catalogue_revision",
        ):
            revision = getattr(self, name)
            if (
                not isinstance(revision, str)
                or len(revision) != 64
                or any(character not in "0123456789abcdef" for character in revision)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 revision")
        if any(
            key.reciprocal_basis_revision != self.reciprocal_basis_revision
            for key in self.prediction.keys
        ):
            raise ValueError("catalogue keys must share the declared reciprocal-basis revision")
        if self.model_id != _PBI2_LANDMARK_MODEL_ID:
            raise ValueError("unsupported PbI2 landmark catalogue model")
        if self.registry_phase_model is not RegistryPhaseModel.FORWARD_H_PLUS_2K:
            raise ValueError("unsupported PbI2 registry phase model")
        expected_revision = _pbi2_landmark_catalogue_revision(
            prediction=self.prediction,
            parent_contributions=contributions,
            selected_parents=selected_parents,
            source_cif_sha256=self.source_cif_sha256,
            reciprocal_basis_revision=self.reciprocal_basis_revision,
            geometry_context_revision=self.geometry_context_revision,
        )
        if self.catalogue_revision != expected_revision:
            raise ValueError("catalogue_revision does not match the declared catalogue payload")
        object.__setattr__(self, "parent_contributions", contributions)
        object.__setattr__(self, "selected_parents", selected_parents)

    @property
    def definitions(self) -> tuple[LayerLMarkerDefinition, ...]:
        return self.prediction.definitions

    @property
    def keys(self) -> tuple[LayerLMarkerKey, ...]:
        return self.prediction.keys


def build_ideal_pbi2_polytype_landmark_catalogue(
    model: ExactTagGeometryModel,
    layer_orders: tuple[CommensurateLayerOrder, ...],
    *,
    parents: tuple[Parent, ...] = PBI2_IDEAL_PARENTS,
) -> Pbi2PolytypeLandmarkCatalogue:
    """Build ideal parent landmarks without evaluating any structure intensity."""

    if not isinstance(model, ExactTagGeometryModel):
        raise TypeError("model must be ExactTagGeometryModel")
    if model.inputs.config.material.phase_id.casefold() != "pbi2":
        raise ValueError("the ideal PbI2 catalogue requires phase_id 'pbi2'")
    motifs = extract_pbi2_motifs(model.inputs.crystal)
    if len(motifs) != 1:
        raise ValueError("the shared-metric catalogue requires exactly one PbI2 trilayer")
    orders = tuple(layer_orders)
    if (
        not orders
        or any(not isinstance(order, CommensurateLayerOrder) for order in orders)
        or len(set(orders)) != len(orders)
        or any(order.denominator not in {1, 2, 3} for order in orders)
    ):
        raise ValueError(
            "PbI2 layer orders must be unique exact values with denominator 1, 2, or 3"
        )
    selected_parents = tuple(Parent(parent) for parent in parents)
    if len(set(selected_parents)) != len(selected_parents) or any(
        parent not in PBI2_IDEAL_PARENTS for parent in selected_parents
    ):
        raise ValueError("parents must be unique ideal PbI2 parents")

    unfiltered = model.enumerate_layer_l_tags(tuple(sorted(orders)))
    kept_indices: list[int] = []
    kept_definitions: list[LayerLMarkerDefinition] = []
    kept_contributions: list[tuple[Pbi2ParentLandmarkContribution, ...]] = []
    for index, definition in enumerate(unfiltered.definitions):
        contributions = tuple(
            item
            for rod_hk in definition.contributing_rod_hk
            for item in pbi2_ideal_parent_landmark_contributions(
                signed_rod_hk=rod_hk,
                layer_order=definition.key.layer_order,
                parents=selected_parents,
            )
        )
        if not contributions:
            continue
        supporting_rods = tuple(sorted({item.signed_rod_hk for item in contributions}))
        kept_indices.append(index)
        kept_definitions.append(
            LayerLMarkerDefinition(
                key=definition.key,
                contributing_rod_hk=supporting_rods,
            )
        )
        kept_contributions.append(contributions)
    if not kept_indices:
        raise ValueError("no requested ideal PbI2 parent landmark reaches the active panel")
    indices = np.asarray(kept_indices, dtype=np.int64)
    prediction = LayerLMarkerPrediction(
        definitions=tuple(kept_definitions),
        coordinates_px=unfiltered.coordinates_px[indices],
        detector_status=unfiltered.detector_status[indices],
        ewald_residual_Ainv=unfiltered.ewald_residual_Ainv[indices],
    )
    catalogue_revision = _pbi2_landmark_catalogue_revision(
        prediction=prediction,
        parent_contributions=tuple(kept_contributions),
        selected_parents=selected_parents,
        source_cif_sha256=model.inputs.config.cif_sha256,
        reciprocal_basis_revision=model.reciprocal_basis_revision,
        geometry_context_revision=model.geometry_context_revision,
    )
    return Pbi2PolytypeLandmarkCatalogue(
        prediction=prediction,
        parent_contributions=tuple(kept_contributions),
        selected_parents=selected_parents,
        source_cif_sha256=model.inputs.config.cif_sha256,
        reciprocal_basis_revision=model.reciprocal_basis_revision,
        geometry_context_revision=model.geometry_context_revision,
        catalogue_revision=catalogue_revision,
    )


__all__ = [
    "PBI2_IDEAL_PARENTS",
    "Pbi2ParentLandmarkContribution",
    "Pbi2PolytypeLandmarkCatalogue",
    "build_ideal_pbi2_polytype_landmark_catalogue",
    "pbi2_ideal_parent_landmark_contributions",
]
