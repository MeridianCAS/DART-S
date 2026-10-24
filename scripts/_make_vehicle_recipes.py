#!/usr/bin/env python3
"""Encode derived vehicle files as recipes against the stock BeamNG files (maintainer tool).

    python scripts/_make_vehicle_recipes.py --beamng-home C:/BeamNG.tech FILE [FILE ...]

Each FILE becomes ``vehicles/recipes/<name>.json`` with output ``vehicles/sbr/<name>``.
A line found in any stock .jbeam/.pc of the source archive becomes a reference, a
near match becomes a character edit that stores only the inserted text, and only a
line with no stock counterpart is stored as text. Encoding fails if such a line
equals or nearly equals a stock line of the audit archives (whitespace removed,
difflib ratio >= 0.8). The summary reports, per file, how the lines were encoded and
the share of the recipe's lines that occur verbatim in the stock files.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_vehicles import (  # noqa: E402
    RECIPE_VERSION,
    RECIPES,
    RecipeError,
    render,
    resolve_home,
    sha256,
)

SOURCE_ARCHIVE = "content/vehicles/sbr.zip"
AUDIT_ARCHIVES = ("content/vehicles/sbr.zip", "content/vehicles/pickup.zip")
STOCK_SUFFIXES = (".jbeam", ".pc")
RELEASE = "BeamNG.tech 0.38.3"
NEAR_RATIO = 0.8
EDIT_RATIO = 0.6
JUMP_COST = 20
LOOKAHEAD = 64


def read_stock(archive: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(archive) as zf:
        return {n: zf.read(n) for n in zf.namelist() if n.endswith(STOCK_SUFFIXES)}


def audit_lines(text: str) -> list[str]:
    """Lines with all whitespace removed, skipping lines made only of brackets and commas."""
    lines = []
    for line in text.splitlines():
        squeezed = "".join(line.split())
        if squeezed.strip("{}[],"):
            lines.append(squeezed)
    return lines


def stock_line_set(stock: dict[str, bytes]) -> set[str]:
    found: set[str] = set()
    for data in stock.values():
        found.update(audit_lines(data.decode("utf-8", "replace")))
    return found


def overlap(text: str, stock_lines: set[str]) -> tuple[int, int]:
    """(lines found verbatim in the stock files, lines counted)."""
    lines = audit_lines(text)
    return sum(1 for line in lines if line in stock_lines), len(lines)


def similarity(a: str, b: str) -> float:
    return max(SequenceMatcher(None, a, b, autojunk=False).ratio(),
               SequenceMatcher(None, b, a, autojunk=False).ratio())


class NearIndex:
    """Best difflib ratio over a set of strings, pruned only by difflib's own upper bounds."""

    def __init__(self, strings):
        self._by_len: dict[int, list[str]] = defaultdict(list)
        for s in set(strings):
            if s:
                self._by_len[len(s)].append(s)
        self._lengths = sorted(self._by_len)
        self._counts: dict[str, Counter] = {}

    def best(self, query: str, floor: float) -> tuple[float, str | None]:
        """Most similar string with ratio >= floor (floor > 0), else (0.0, None)."""
        lq = len(query)
        if not lq:
            return 0.0, None
        lo, hi = math.ceil(lq * floor / (2 - floor)), math.floor(lq * (2 - floor) / floor)
        lengths = sorted((n for n in self._lengths if lo <= n <= hi), key=lambda n: abs(n - lq))
        wanted = Counter(query)
        best_r, best_s = floor, None
        for n in lengths:
            if 2 * min(lq, n) / (lq + n) < best_r:
                continue
            for s in self._by_len[n]:
                have = self._counts.get(s)
                if have is None:
                    have = self._counts[s] = Counter(s)
                shared = sum(min(k, have[ch]) for ch, k in wanted.items())
                if 2 * shared / (lq + n) < best_r:
                    continue
                r = similarity(query, s)
                if r > best_r or (best_s is None and r >= best_r):
                    best_r, best_s = r, s
        return (best_r, best_s) if best_s is not None else (0.0, None)


def literal_index(stock: dict[str, bytes]) -> NearIndex:
    """Index for the stored-text guard: stock lines with whitespace removed."""
    lines = set()
    for data in stock.values():
        lines.update("".join(line.split()) for line in data.decode("utf-8", "replace").splitlines())
    return NearIndex(lines)


def near_stock_literals(recipe: dict, guard: NearIndex) -> list[tuple[str, float, str]]:
    """Stored text lines of a recipe that equal or nearly equal a stock line."""
    hits = []
    for op in recipe["ops"]:
        if op[0] == "L":
            r, s = guard.best("".join(op[1].split()), NEAR_RATIO)
            if s is not None:
                hits.append((op[1], r, s))
    return hits


class Stock:
    """Stock entries of the source archive, indexed by whitespace-stripped line."""

    def __init__(self, entries: dict[str, bytes]):
        self.names = sorted(entries)
        self.digest = {n: sha256(entries[n]) for n in self.names}
        self.lines = {n: entries[n].decode("utf-8").split("\n") for n in self.names}
        self.where: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for n in self.names:
            for i, line in enumerate(self.lines[n]):
                if line.strip():
                    self.where[line.strip()].append((n, i))
        self.texts = NearIndex(self.where)

    def text(self, at: tuple[str, int] | None) -> str | None:
        if at is None or not 0 <= at[1] < len(self.lines[at[0]]):
            return None
        return self.lines[at[0]][at[1]]


class EncodeError(RuntimeError):
    pass


def _lead(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _trail(line: str) -> str:
    return line[len(line.rstrip()):]


def _allowed(mine: str, stock: str) -> set:
    return {mine, None} if mine == stock else {mine}


def edit_list(base: str, text: str) -> list[list]:
    sm = SequenceMatcher(None, base, text, autojunk=False)
    return [[i1, i2, text[j1:j2]] for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


def pick_newline(text: str) -> str:
    for sep in ("\r\n", "\n\n"):
        if sep in text and all("\n" not in part for part in text.split(sep)):
            return sep
    return "\n"


def encode(data: bytes, output: str, stock: Stock, guard: NearIndex, *,
           archive: str = SOURCE_ARCHIVE, release: str = RELEASE,
           edit_ratio: float = EDIT_RATIO) -> dict:
    text = data.decode("utf-8")
    newline = pick_newline(text)
    lines = text.split(newline)
    items: list[tuple] = []
    cur: tuple[str, int] | None = None

    def run_length(at: tuple[str, int], k: int) -> int:
        src, n = stock.lines[at[0]], 0
        while (n < LOOKAHEAD and k + 1 + n < len(lines) and at[1] + 1 + n < len(src)
               and lines[k + 1 + n].strip() == src[at[1] + 1 + n].strip()):
            n += 1
        return n

    def nearest(cands: list[tuple[str, int]], k: int | None) -> tuple[str, int]:
        def key(c):
            same = cur is not None and c[0] == cur[0]
            return (run_length(c, k) if k is not None else 0, same, -abs(c[1] - cur[1] - 1) if same else 0)
        return max(cands, key=key)

    for k, line in enumerate(lines):
        core = line.strip()
        nxt = (cur[0], cur[1] + 1) if cur else None
        nxt_line = stock.text(nxt)
        if not core:
            if nxt_line is not None and not nxt_line.strip():
                items.append(("ref", nxt, {"blank": _allowed(line, nxt_line)}))
                cur = nxt
            else:
                items.append(("ws", line))
            continue
        if nxt_line is not None and nxt_line.strip() == core:
            at = nxt
        elif core in stock.where:
            at = nearest(stock.where[core], k)
        else:
            at = None
        if at is not None:
            s = stock.text(at)
            items.append(("ref", at, {"lead": _allowed(_lead(line), _lead(s)),
                                      "trail": _allowed(_trail(line), _trail(s))}))
            cur = at
            continue
        # Edit the line that follows the previous one unless another stock line gives a
        # clearly shorter edit list (the jump costs a new reference op later).
        options = []
        if nxt_line and nxt_line.strip() and similarity(nxt_line.strip(), core) >= edit_ratio:
            edits = edit_list(nxt_line.strip(), core)
            options.append((len(json.dumps(edits)), 0, nxt, edits))
        r, s = stock.texts.best(core, edit_ratio)
        if s is not None:
            edits = edit_list(s, core)
            options.append((len(json.dumps(edits)) + JUMP_COST, 1, nearest(stock.where[s], None), edits))
        if options:
            _, _, at, edits = min(options, key=lambda o: o[:2])
            base = stock.text(at)
            lead, trail = _lead(line), _trail(line)
            items.append(("edit", at, None if lead == _lead(base) else lead,
                          None if trail == _trail(base) else trail, edits))
            cur = at
            continue
        near_r, near_s = guard.best("".join(core.split()), NEAR_RATIO)
        if near_s is not None:
            raise EncodeError(f"{output}: line {k + 1} would be stored as text but is close to the "
                              f"stock line {near_s[:80]!r} (ratio {near_r:.2f})")
        items.append(("lit", line))

    sources: list[str] = []

    def src_index(name: str) -> int:
        if name not in sources:
            sources.append(name)
        return sources.index(name)

    ops: list[list] = []
    pending: dict | None = None

    def flush() -> None:
        nonlocal pending
        if pending:
            ws = []
            for field in ("lead", "trail", "blank"):
                allowed = pending["cons"].get(field)
                ws.append(None if allowed is None or None in allowed else next(iter(allowed)))
            ops.append(["R", pending["src"], pending["start"], pending["count"], *ws])
        pending = None

    for item in items:
        kind = item[0]
        if kind == "ref":
            (name, idx), cons = item[1], item[2]
            src = src_index(name)
            if (pending and pending["src"] == src and pending["start"] + pending["count"] == idx
                    and all(not (f in pending["cons"] and pending["cons"][f].isdisjoint(v)) for f, v in cons.items())):
                for f, v in cons.items():
                    pending["cons"][f] = pending["cons"][f] & v if f in pending["cons"] else set(v)
                pending["count"] += 1
            else:
                flush()
                pending = {"src": src, "start": idx, "count": 1, "cons": {f: set(v) for f, v in cons.items()}}
            continue
        flush()
        if kind == "edit":
            _, (name, idx), lead, trail, edits = item
            ops.append(["E", src_index(name), idx, lead, trail, edits])
        elif kind == "ws":
            if ops and ops[-1][0] == "W" and ops[-1][1] == item[1]:
                ops[-1][2] += 1
            else:
                ops.append(["W", item[1], 1])
        else:
            ops.append(["L", item[1]])
    flush()

    recipe = {
        "recipe_version": RECIPE_VERSION,
        "output": output,
        "output_sha256": sha256(data),
        "beamng": release,
        "archive": archive,
        "sources": [[name, stock.digest[name]] for name in sources],
        "newline": newline,
        "ops": ops,
    }
    if render(recipe, [stock.lines[n] for n in sources]) != data:
        raise EncodeError(f"{output}: recipe does not reproduce the input")
    return recipe


def dump_recipe(recipe: dict) -> str:
    """JSON with one source and one op per line, so recipe diffs stay readable."""
    head = [f" {json.dumps(k)}: {json.dumps(v)}" for k, v in recipe.items() if k not in ("sources", "ops")]
    sources = ",\n".join(f"  {json.dumps(s)}" for s in recipe["sources"])
    ops = ",\n".join(f"  {json.dumps(op)}" for op in recipe["ops"])
    return "{\n" + ",\n".join(head) + f',\n "sources": [\n{sources}\n ],\n "ops": [\n{ops}\n ]\n}}\n'


def recipe_stats(recipe: dict) -> dict:
    ops = recipe["ops"]
    return {
        "ref_ops": sum(op[0] == "R" for op in ops),
        "ref_lines": sum(op[3] for op in ops if op[0] == "R"),
        "edits": sum(op[0] == "E" for op in ops),
        "inserted_chars": sum(len(e[2]) for op in ops if op[0] == "E" for e in op[5]),
        "longest_insert": max((len(e[2]) for op in ops if op[0] == "E" for e in op[5]), default=0),
        "whitespace_lines": sum(op[2] for op in ops if op[0] == "W"),
        "literal_lines": sum(op[0] == "L" for op in ops),
        "literal_chars": sum(len(op[1].strip()) for op in ops if op[0] == "L"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", type=Path, help="derived vehicle files to encode")
    ap.add_argument("--beamng-home", help="BeamNG.tech install used as the source")
    ap.add_argument("--output-dir", default="vehicles/sbr", help="repo-relative directory of the generated files")
    ap.add_argument("--recipes-dir", type=Path, default=RECIPES)
    ap.add_argument("--release", default=RELEASE, help="BeamNG.tech release recorded in the recipes")
    args = ap.parse_args(argv)

    home = resolve_home(args.beamng_home)
    stock = Stock(read_stock(home / SOURCE_ARCHIVE))
    audit = {}
    for rel in AUDIT_ARCHIVES:
        if (home / rel).is_file():
            audit.update({f"{rel}:{n}": d for n, d in read_stock(home / rel).items()})
        else:
            print(f"note: {home / rel} not found; audit uses the remaining archives")
    stock_lines = stock_line_set(audit)
    guard = literal_index(audit)

    args.recipes_dir.mkdir(parents=True, exist_ok=True)
    worst = 0.0
    for path in args.files:
        data = path.read_bytes()
        try:
            recipe = encode(data, f"{args.output_dir.rstrip('/')}/{path.name}", stock, guard, release=args.release)
        except (EncodeError, RecipeError, UnicodeDecodeError) as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
            return 1
        body = dump_recipe(recipe)
        out = args.recipes_dir / f"{path.name}.json"
        out.write_text(body, encoding="utf-8", newline="\n")
        st = recipe_stats(recipe)
        hit, total = overlap(body, stock_lines)
        src_hit, src_total = overlap(data.decode("utf-8"), stock_lines)
        near = near_stock_literals(recipe, guard)
        worst = max(worst, hit / total if total else 0.0)
        print(f"{path.name}: {len(data)} B -> {out.name} {len(body.encode())} B | "
              f"R {st['ref_ops']} ops/{st['ref_lines']} lines, E {st['edits']} "
              f"(+{st['inserted_chars']} chars, longest {st['longest_insert']}), "
              f"W {st['whitespace_lines']}, L {st['literal_lines']} ({st['literal_chars']} chars, "
              f"{len(near)} near stock) | stock overlap: input {src_hit}/{src_total}, recipe {hit}/{total}")
    print(f"max recipe overlap {worst:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
