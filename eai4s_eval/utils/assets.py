from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class AssetResolutionError(FileNotFoundError):
    """Raised when a required local USD asset cannot be resolved."""


@dataclass(frozen=True)
class PickLiftAssetPaths:
    shelf: str
    tube: str
    beaker: str
    vial: str
    boiling_flask: str
    erlenmeyer_flask: str


@dataclass(frozen=True)
class TubeRackInsertAssetPaths:
    tube: str
    rack: str
    beaker: str
    vial: str
    boiling_flask: str
    erlenmeyer_flask: str


def _candidate_roots(asset_root: str | None) -> list[Path]:
    here = Path(__file__).resolve()
    code_dir = here.parents[2]
    project_dir = here.parents[3]
    roots: list[Path] = []
    if asset_root:
        roots.append(Path(asset_root).expanduser())
    roots.extend(
        [
            Path.cwd(),
            code_dir,
            code_dir / "assets",
            project_dir,
            project_dir / "assets",
        ]
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve() if root.exists() else root
        if resolved not in seen:
            unique.append(root)
            seen.add(resolved)
    return unique


def resolve_asset_path(relative_path: str, asset_root: str | None = None, must_exist: bool = True) -> str:
    """Resolve a local asset path while keeping Isaac Nucleus paths untouched."""

    path = Path(relative_path)
    if path.is_absolute():
        if path.exists() or not must_exist:
            return str(path)
        raise AssetResolutionError(f"Asset path does not exist: {path}")

    searched: list[str] = []
    for root in _candidate_roots(asset_root):
        candidate = root / relative_path
        searched.append(str(candidate))
        if candidate.exists():
            return str(candidate)

    if not must_exist:
        root = _candidate_roots(asset_root)[0]
        return str(root / relative_path)

    searched_text = "\n  - ".join(searched)
    raise AssetResolutionError(
        "Could not resolve required USD asset. Pass --asset_root to the directory "
        f"that contains the assets folder.\nMissing: {relative_path}\nSearched:\n  - {searched_text}"
    )


def pick_lift_asset_paths(asset_root: str | None = None, must_exist: bool = True) -> PickLiftAssetPaths:
    return PickLiftAssetPaths(
        shelf=resolve_asset_path("assets/shelf/shelf.usd", asset_root, must_exist),
        tube=resolve_asset_path("assets/tube/test_tube.usd", asset_root, must_exist),
        beaker=resolve_asset_path("assets/beaker/beaker.usd", asset_root, must_exist),
        vial=resolve_asset_path("assets/vial/vial_brown/vial_brown.usd", asset_root, must_exist),
        boiling_flask=resolve_asset_path("assets/boiling_flask/flask.usd", asset_root, must_exist),
        erlenmeyer_flask=resolve_asset_path("assets/Erlenmeyer_flask/flask.usd", asset_root, must_exist),
    )


def tube_rack_insert_asset_paths(asset_root: str | None = None, must_exist: bool = True) -> TubeRackInsertAssetPaths:
    return TubeRackInsertAssetPaths(
        tube=resolve_asset_path("assets/test-tube/test_tube.usd", asset_root, must_exist),
        rack=resolve_asset_path("assets/tube-rack/tube_rack.usd", asset_root, must_exist),
        beaker=resolve_asset_path("assets/beaker/beaker.usd", asset_root, must_exist),
        vial=resolve_asset_path("assets/vial/vial_brown/vial_brown.usd", asset_root, must_exist),
        boiling_flask=resolve_asset_path("assets/boiling_flask/flask.usd", asset_root, must_exist),
        erlenmeyer_flask=resolve_asset_path("assets/Erlenmeyer_flask/flask.usd", asset_root, must_exist),
    )
