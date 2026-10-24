"""Tests for the vehicle recipes: synthetic round trips, plus a full build when BeamNG.tech is installed."""
from __future__ import annotations

import copy
import json
import os
import zipfile
from pathlib import Path

import pytest

from scripts import _make_vehicle_recipes as mk
from scripts import build_vehicles as bv

REPO = Path(__file__).resolve().parent.parent
ARCHIVE = "content/vehicles/sbr.zip"
HOME = bv.resolve_home()

BODY_LINES = [
    "{",
    '"demo_body": {',
    '    "information":{',
    '        "authors":"Example Author",',
    '        "name":"Demo Body",',
    "    },",
    "",
    '    "slotType" : "demo_body",',
    '    "nodes": [',
    '        ["id", "posX", "posY", "posZ"],',
    '        {"nodeWeight":4.5},',
    '        ["b1",    0.10, -1.20, 0.30],',
    '        ["b2", 0.40, -1.20, 0.30, {"group":"demo_body"}],',
    '        ["b3", 0.70, -1.10, 0.35],',
    '        ["b4", -0.70, 1.10, 0.35],',
    "    ],",
    "",
    '    "beams": [',
    '        ["id1:", "id2:"],',
    '        ["b1", "b2"],',
    '        ["b2", "b3"],',
    '        ["b3", "b4"],',
    "    ],",
    "}",
    "}",
]
PC_LINES = ["{", '"format":2,', '"model":"demo",', '"parts":{', '  "demo_body":"demo_body",',
            '  "demo_motor":"demo_motor_200"', "}", "}"]
STOCK = {
    "vehicles/demo/demo_body.jbeam": "\r\n".join(BODY_LINES).encode(),
    "vehicles/demo/demo.pc": "\r\n".join(PC_LINES).encode(),
}


def derived_variants() -> dict[str, bytes]:
    crlf = list(BODY_LINES)
    crlf[1] = '"demo_body_heavy": {'
    crlf[13] = '        ["b3", 0.70, -1.10, 0.35, {"nodeWeight":6.25}],'
    collapsed = [(" " + " ".join(line.split())) if line[:1] == " " else line for line in BODY_LINES]
    moved = list(BODY_LINES)
    moved[19], moved[21] = moved[21], moved[19]
    moved.insert(16, moved.pop(7))
    literal = list(BODY_LINES)
    literal.insert(22, '        ["b1", "b4", {"inputSource":"dartsSuspFL", "inRate":2.0}],')
    pc_spaced = [line.replace('":"', '": "').replace('":2', '": 2') for line in PC_LINES]
    return {
        "crlf_numeric.jbeam": "\r\n".join(crlf).encode(),
        "lf_collapsed.jbeam": "\n".join(collapsed).encode(),
        "moved.jbeam": "\n".join(moved).encode(),
        "gapped.jbeam": "\n\n".join(BODY_LINES).encode(),
        "literal.jbeam": ("\n".join(literal) + "\n").encode(),
        "pc_spaced.pc": "\n".join(pc_spaced).encode(),
    }


def write_home(root: Path, entries: dict[str, bytes]) -> Path:
    archive = root / ARCHIVE
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return root


@pytest.fixture(scope="module")
def synthetic():
    stock = mk.Stock(STOCK)
    guard = mk.literal_index(STOCK)
    recipes = {name: mk.encode(data, f"vehicles/sbr/{name}", stock, guard)
               for name, data in derived_variants().items()}
    return stock, guard, recipes


def _sources(stock, recipe):
    return [stock.lines[name] for name, _ in recipe["sources"]]


def test_round_trip_is_byte_identical(synthetic):
    stock, _, recipes = synthetic
    for name, data in derived_variants().items():
        recipe = json.loads(mk.dump_recipe(recipes[name]))
        assert bv.render(recipe, _sources(stock, recipe)) == data, name
        assert recipe["output_sha256"] == bv.sha256(data)


def test_encoding_stores_only_differences(synthetic):
    _, _, r = synthetic
    assert r["crlf_numeric.jbeam"]["newline"] == "\r\n"
    assert r["gapped.jbeam"]["newline"] == "\n\n"
    inserted = [e[2] for op in r["crlf_numeric.jbeam"]["ops"] if op[0] == "E" for e in op[5]]
    assert "_heavy" in inserted
    assert any("6.25" in text for text in inserted)
    for name in ("lf_collapsed.jbeam", "moved.jbeam", "gapped.jbeam", "pc_spaced.pc"):
        ops = r[name]["ops"]
        assert not [op for op in ops if op[0] == "L"], name
        assert not [e for op in ops if op[0] == "E" for e in op[5] if e[2].strip()], name
    literals = [op[1] for op in r["literal.jbeam"]["ops"] if op[0] == "L"]
    assert literals == ['        ["b1", "b4", {"inputSource":"dartsSuspFL", "inRate":2.0}],']


def test_build_check_and_rerun(synthetic, tmp_path):
    _, _, recipes = synthetic
    recipes = list(recipes.values())
    home = write_home(tmp_path / "bng", STOCK)
    out = tmp_path / "out"
    assert all(changed for _, changed in bv.build(recipes, home, out))
    for name, data in derived_variants().items():
        assert (out / name).read_bytes() == data
    assert bv.check(recipes, out) == []
    assert not any(changed for _, changed in bv.build(recipes, home, out))
    (out / "moved.jbeam").write_bytes(b"{}")
    (out / "gapped.jbeam").unlink()
    assert sorted(p.split()[0] for p in bv.check(recipes, out)) == ["differs", "missing"]


def test_source_mismatch_is_rejected(synthetic, tmp_path):
    _, _, recipes = synthetic
    changed = dict(STOCK)
    changed["vehicles/demo/demo_body.jbeam"] = STOCK["vehicles/demo/demo_body.jbeam"].replace(b"0.10", b"0.11")
    home = write_home(tmp_path, changed)
    with pytest.raises(bv.RecipeError, match="pinned sha256.*BeamNG.tech 0.38.3"):
        bv.build(list(recipes.values()), home, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_output_mismatch_is_rejected(synthetic, tmp_path):
    _, _, recipes = synthetic
    recipe = copy.deepcopy(recipes["crlf_numeric.jbeam"])
    edit = next(op for op in recipe["ops"] if op[0] == "E")
    edit[5][-1][2] += "0"
    with pytest.raises(bv.RecipeError, match="do not match the pinned sha256"):
        bv.build([recipe], write_home(tmp_path, STOCK), tmp_path / "out")


def test_whitespace_fields_must_be_whitespace(synthetic):
    stock, _, recipes = synthetic
    recipe = copy.deepcopy(recipes["lf_collapsed.jbeam"])
    ref = next(op for op in recipe["ops"] if op[0] == "R" and op[4] is not None)
    ref[4] = " x"
    with pytest.raises(bv.RecipeError):
        bv.render(recipe, _sources(stock, recipe))


def test_text_lines_close_to_stock_are_refused(synthetic):
    stock, guard, recipes = synthetic
    near = list(BODY_LINES)
    near[12] = '        ["b2", 0.41, -1.20, 0.30, {"group":"demo_body"}],'
    with pytest.raises(mk.EncodeError, match="close to the stock line"):
        mk.encode("\n".join(near).encode(), "vehicles/sbr/near.jbeam", stock, guard, edit_ratio=1.01)
    recipe = copy.deepcopy(recipes["literal.jbeam"])
    recipe["ops"].append(["L", BODY_LINES[13].replace("0.35", "0.36")])
    assert len(mk.near_stock_literals(recipe, guard)) == 1
    assert mk.near_stock_literals(recipes["literal.jbeam"], guard) == []


needs_beamng = pytest.mark.skipif(not (HOME / ARCHIVE).is_file(), reason=f"BeamNG.tech not found at {HOME}")


@pytest.fixture(scope="module")
def installed():
    paths = sorted(bv.RECIPES.glob("*.json"))
    audit = {}
    for rel in mk.AUDIT_ARCHIVES:
        if (HOME / rel).is_file():
            audit.update({f"{rel}:{n}": d for n, d in mk.read_stock(HOME / rel).items()})
    return paths, [bv.load_recipe(p) for p in paths], mk.stock_line_set(audit), mk.literal_index(audit)


@needs_beamng
def test_recipes_rebuild_pinned_outputs(installed, tmp_path):
    _, recipes, _, _ = installed
    assert recipes
    bv.build(recipes, HOME, tmp_path)
    for recipe in recipes:
        data = (tmp_path / Path(recipe["output"]).name).read_bytes()
        assert bv.sha256(data) == recipe["output_sha256"], recipe["output"]


@needs_beamng
def test_shipped_files_share_under_10pct_of_lines_with_stock(installed):
    paths, recipes, stock_lines, _ = installed
    generated = {REPO / r["output"] for r in recipes}
    shipped = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d != ".git"]
        shipped += [Path(root) / f for f in files if f.endswith(mk.STOCK_SUFFIXES)]
    for path in [p for p in shipped if p not in generated] + paths:
        hit, total = mk.overlap(path.read_text(encoding="utf-8"), stock_lines)
        assert hit < (0.10 * total if total else 1), path.relative_to(REPO)


@needs_beamng
def test_recipe_text_lines_are_not_near_stock(installed):
    paths, recipes, _, guard = installed
    for path, recipe in zip(paths, recipes):
        assert mk.near_stock_literals(recipe, guard) == [], path.name
