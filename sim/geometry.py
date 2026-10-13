"""Simple box geometry (spec 6.2)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from sim.home import HomeState
from sim.parameters import params


@dataclass
class Geometry:
    storeys: int
    footprint: float
    perimeter: float
    exposure: float
    wall_area_gross: float
    window_area: float
    wall_area_net: float
    roof_area: float
    roof_exposure: float
    floor_ground_area: float
    floor_exposure: float
    volume: float

    @property
    def envelope_area(self) -> float:
        return (
            self.wall_area_net
            + self.window_area
            + self.roof_area * self.roof_exposure
            + self.floor_ground_area * self.floor_exposure
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["envelope_area"] = self.envelope_area
        return d


def flat_exposures(position: str, default: float) -> tuple[float, float]:
    """(roof_exposure, floor_exposure) for a flat."""
    return {
        "ground": (0.0, 1.0),
        "middle": (0.0, 0.0),
        "top": (1.0, 0.0),
    }.get(position, (default, default))


def compute_geometry(state: HomeState) -> Geometry:
    p = params()
    storeys = max(1, int(state.storeys))
    h = p["storey_height_m"]
    footprint = state.floor_area / storeys
    perimeter = 4.0 * math.sqrt(footprint)
    exposure = p["exposure"][state.property_type]
    wall_gross = perimeter * h * storeys * exposure
    window = p["glazing_ratio"] * state.floor_area
    wall_net = max(wall_gross - window, 0.0)
    roof_area = footprint * (p["pitched_roof_area_factor"] if state.roof_type == "pitched" else 1.0)
    if state.property_type == "flat":
        roof_exp, floor_exp = flat_exposures(state.flat_position, p["flat_roof_floor_exposure"])
    else:
        roof_exp, floor_exp = 1.0, 1.0
    return Geometry(
        storeys=storeys,
        footprint=footprint,
        perimeter=perimeter,
        exposure=exposure,
        wall_area_gross=wall_gross,
        window_area=window,
        wall_area_net=wall_net,
        roof_area=roof_area,
        roof_exposure=roof_exp,
        floor_ground_area=footprint,
        floor_exposure=floor_exp,
        volume=state.floor_area * h,
    )


def usable_roof_area(state: HomeState, geom: Geometry) -> float:
    """Free roof area for PV: the customer's figure, else one pitch (or 60% of a flat roof)."""
    if state.roof_free_area_m2 is not None:
        return float(state.roof_free_area_m2)
    if geom.roof_exposure == 0.0:
        return 0.0
    share = 0.45 if state.roof_type == "pitched" else 0.6
    return geom.roof_area * share * geom.roof_exposure
