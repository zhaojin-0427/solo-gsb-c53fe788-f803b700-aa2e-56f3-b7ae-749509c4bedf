"""Generate minimal valid instances and validate instances against nodes.

The generator produces the *smallest* value satisfying a normalized node
(fewer object keys, shorter strings/arrays), and follows reference cycles with
a depth budget so recursive schemas terminate. The validator is the same
constraint set the comparator reasons about; it is used to verify that a
generated counter-example passes exactly one side.

Call coverage additionally needs a *tri-state* verdict (see
:func:`coverage_verdict`): every value/node check is classified as
definitely-passing (``PASS``), definitely-failing (``FAIL``) or undecided
(``PENDING`` — the verdict hinges on an unsupported keyword, an unresolvable
``$ref`` or a reference cycle). PENDING is what lets ``oneOf``/``anyOf`` be
adjudicated per their real semantics instead of optimistically covered.
"""
from __future__ import annotations

from typing import Any

from .schemagraph import SchemaNode, UNSET

MAX_DEPTH = 8

# Tri-state coverage verdicts
PASS = "pass"      # all decidable constraints satisfied, no pending facets
FAIL = "fail"      # a known constraint is violated
PENDING = "pending"  # not known to fail, but unsupported constraints decide


class ValidationError(Exception):
    def __init__(self, message: str, path: str = ""):
        super().__init__(message)
        self.message = message
        self.path = path


def _json_type(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "unknown"


def type_allows(node: SchemaNode, value_type: str, value: Any = None) -> bool:
    if node.type is None:
        return True
    allowed = set(node.type)
    if node.nullable:
        allowed.add("null")
    if value_type in allowed:
        return True
    # an integer also satisfies number
    if value_type == "integer" and "number" in allowed:
        return True
    return False


def validate(node: SchemaNode, value: Any, path: str = "",
             depth: int = 0) -> list[str]:
    """Return a list of validation error messages (empty => valid).

    This validator mirrors the constraint set the comparator reasons about;
    it is deliberately lenient on ``oneOf`` (treats it like ``anyOf``) and
    optimistic on cycles/unsupported keywords — the strict tri-state
    semantics needed for call coverage live in :func:`coverage_verdict`.
    """
    errors: list[str] = []
    if node.is_bool is True:
        return errors
    if node.is_bool is False:
        return [f"{path or '#'}: false 模式不允许任何值"]
    if node.is_cycle:
        return errors  # cannot judge => optimistically accept
    if node.invalid:
        return [f"{path or '#'}: 模式不可判定（{node.invalid}）"]
    if node.unsupported:
        # We can still check the facets we know; unsupported keywords are
        # treated as transparent for replay purposes, and reported separately.
        pass

    vt = _json_type(value)
    if not type_allows(node, vt, value):
        errors.append(f"{path or '#'}: 类型 {vt} 不被允许 {sorted(node.type or [])}")
        return errors  # further facet checks would be noise

    if value is None:
        return errors

    errors.extend(_facet_errors(node, value, path))

    if vt == "array" and node.items is not None and depth < MAX_DEPTH:
        for i, item in enumerate(value):
            errors.extend(validate(node.items, item,
                                   f"{path}/{i}", depth + 1))

    if vt == "object":
        for pname, pnode in node.properties.items():
            if pname in value and depth < MAX_DEPTH:
                # request/response filtering at replay time is done by the
                # caller; raw nodes keep both readOnly/writeOnly facets
                errors.extend(validate(pnode, value[pname],
                                       f"{path}/{_esc(pname)}", depth + 1))
        extra = {k: v for k, v in value.items() if k not in node.properties}
        if isinstance(node.additional, SchemaNode) and depth < MAX_DEPTH:
            for k, v in sorted(extra.items()):
                errors.extend(validate(node.additional, v,
                                       f"{path}/{_esc(k)}", depth + 1))

        if node.one_of and depth < MAX_DEPTH:
            if not any(not validate(b, value, path, depth + 1)
                       for b in node.one_of):
                errors.append(f"{path or '#'}: 不满足 oneOf 的任何分支")
        if node.any_of and depth < MAX_DEPTH:
            if not any(not validate(b, value, path, depth + 1)
                       for b in node.any_of):
                errors.append(f"{path or '#'}: 不满足 anyOf 的任何分支")
    return errors


def _facet_errors(node: SchemaNode, value: Any, path: str = "") -> list[str]:
    """Non-recursive, known-constraint checks of one node (type pre-checked)."""
    errors: list[str] = []
    vt = _json_type(value)

    if node.enum is not None:
        if not any(_json_safe_equal(value, e) for e in node.enum):
            errors.append(f"{path or '#'}: 值不在枚举 {node.enum} 中")
    if node.const is not UNSET and node.const is not None:
        if not _json_safe_equal(value, node.const):
            errors.append(f"{path or '#'}: 值必须等于常量 {node.const!r}")

    if vt in ("integer", "number"):
        if node.minimum is not None and value < node.minimum:
            errors.append(f"{path or '#'}: {value} 小于 minimum {node.minimum}")
        if node.maximum is not None and value > node.maximum:
            errors.append(f"{path or '#'}: {value} 大于 maximum {node.maximum}")
        if node.ex_min is not None and value <= node.ex_min:
            errors.append(
                f"{path or '#'}: {value} 必须大于 exclusiveMinimum {node.ex_min}")
        if node.ex_max is not None and value >= node.ex_max:
            errors.append(
                f"{path or '#'}: {value} 必须小于 exclusiveMaximum {node.ex_max}")
        if node.multiple_of is not None and not _is_multiple(value,
                                                             node.multiple_of):
            errors.append(
                f"{path or '#'}: {value} 不是 {node.multiple_of} 的倍数")

    if vt == "string":
        if node.min_length is not None and len(value) < node.min_length:
            errors.append(
                f"{path or '#'}: 字符串长度小于 minLength {node.min_length}")
        if node.max_length is not None and len(value) > node.max_length:
            errors.append(
                f"{path or '#'}: 字符串长度超过 maxLength {node.max_length}")

    if vt == "array":
        if node.min_items is not None and len(value) < node.min_items:
            errors.append(
                f"{path or '#'}: 数组元素少于 minItems {node.min_items}")
        if node.max_items is not None and len(value) > node.max_items:
            errors.append(
                f"{path or '#'}: 数组元素超过 maxItems {node.max_items}")
        if node.unique_items:
            seen = [_json_key(v) for v in value]
            if len(set(seen)) != len(seen):
                errors.append(f"{path or '#'}: 数组元素不唯一")

    if vt == "object":
        if node.min_props is not None and len(value) < node.min_props:
            errors.append(
                f"{path or '#'}: 属性数少于 minProperties {node.min_props}")
        if node.max_props is not None and len(value) > node.max_props:
            errors.append(
                f"{path or '#'}: 属性数超过 maxProperties {node.max_props}")
        for req in node.required:
            if req not in value:
                errors.append(f"{path or '#'}: 缺少必填属性 {req!r}")
        extra = {k: v for k, v in value.items() if k not in node.properties}
        if node.additional is False and extra:
            errors.append(
                f"{path or '#'}: 不允许额外属性 {sorted(extra)}")
    return errors


# Unsupported keyword -> JSON types it constrains. Keywords missing from this
# map (``not``, ``if``/``then``/``else``, unknown keywords, ``type[]`` …) may
# constrain any value type.
_UNSUPPORTED_KW_TYPES = {
    "pattern": {"string"},
    "contains": {"array"},
    "minContains": {"array"},
    "maxContains": {"array"},
    "prefixItems": {"array"},
    "unevaluatedItems": {"array"},
    "propertyNames": {"object"},
    "patternProperties": {"object"},
    "dependentRequired": {"object"},
    "dependentSchemas": {"object"},
    "unevaluatedProperties": {"object"},
}


def coverage_verdict(node: SchemaNode, value: Any, path: str = "",
                     depth: int = 0) -> dict:
    """Tri-state verdict of ``value`` against ``node`` for call coverage.

    Walks the parsed schema graph *along the actual value* (local ``$ref``
    already resolved into the graph, nested properties, array items and
    ``oneOf``/``anyOf`` branches) and returns a dict::

        {"state": PASS | FAIL | PENDING,
         "hits":  [(keyword, FacetLoc), ...]   # unsupported keyword facets
         "notes": [(text, FacetLoc | None), ...],  # unjudgeable sites
         "errors": [str, ...]}                 # known-constraint violations

    Branch semantics are exact: each ``oneOf``/``anyOf`` member is itself
    classified into the three states, then the combination is adjudicated per
    the JSON Schema meaning of "exactly one" / "at least one":

    * oneOf: exactly one PASS  => PASS;
      two or more PASS          => FAIL (the "exactly one" contract breaks);
      no PASS and any PENDING   => PENDING (cannot decide, pending keywords
                                  and their locations are reported).
    * anyOf: at least one PASS  => PASS (a cleanly validated branch settles
               the verdict even if other branches stay pending);
      no PASS and any PENDING   => PENDING;
      every branch FAIL         => FAIL.

    Optional properties absent from the value are never visited, and keywords
    that do not apply to the value's JSON type (e.g. ``pattern`` on an
    integer) do not make the verdict pending.
    """
    hits: list = []
    notes: list = []
    errors: list[str] = []

    def pending():
        return {"state": PENDING, "hits": hits, "notes": notes,
                "errors": errors}

    if node is None or depth >= MAX_DEPTH:
        return {"state": PASS, "hits": hits, "notes": notes,
                "errors": errors}
    if node.is_bool is True:
        return {"state": PASS, "hits": hits, "notes": notes,
                "errors": errors}
    if node.is_bool is False:
        errors.append(f"{path or '#'}: false 模式不允许任何值")
        return {"state": FAIL, "hits": hits, "notes": notes,
                "errors": errors}
    if node.is_cycle:
        notes.append(("该取值的约束涉及循环引用，无法判定", node.loc))
        return pending()
    if node.invalid:
        notes.append((f"该取值的约束无法判定（{node.invalid}）", node.loc))
        return pending()

    vt = _json_type(value)

    # unsupported keywords at this node that apply to the value's JSON type
    for kw, loc in node.unsupported:
        applies = _UNSUPPORTED_KW_TYPES.get(kw)
        if applies is None or vt in applies:
            hits.append((kw, loc))

    if not type_allows(node, vt, value):
        errors.append(
            f"{path or '#'}: 类型 {vt} 不被允许 {sorted(node.type or [])}")
        return {"state": FAIL, "hits": [], "notes": [], "errors": errors}

    if value is None:
        # only unsupported keywords reachable for null (e.g. not, if/then)
        return pending() if hits else \
            {"state": PASS, "hits": hits, "notes": notes, "errors": errors}

    local_errors = _facet_errors(node, value, path)
    if local_errors:
        return {"state": FAIL, "hits": [], "notes": [],
                "errors": local_errors}

    # recurse along the actual value; absent optional properties are skipped
    if vt == "array" and node.items is not None and depth < MAX_DEPTH:
        for i, item in enumerate(value):
            sub = coverage_verdict(node.items, item, f"{path}/{i}", depth + 1)
            _merge_child(sub, hits, notes, errors)

    if vt == "object" and depth < MAX_DEPTH:
        for pname, pnode in node.properties.items():
            if pname in value:
                sub = coverage_verdict(
                    pnode, value[pname], f"{path}/{_esc(pname)}", depth + 1)
                _merge_child(sub, hits, notes, errors)
        if isinstance(node.additional, SchemaNode):
            extra = [k for k in value if k not in node.properties]
            for k in sorted(extra):
                sub = coverage_verdict(
                    node.additional, value[k], f"{path}/{_esc(k)}", depth + 1)
                _merge_child(sub, hits, notes, errors)

    # node-level properties/items are ANDed with any composition: a known
    # violation already settles the verdict before branches are adjudicated
    if errors:
        return {"state": FAIL, "hits": [], "notes": [], "errors": errors}

    # compositions: exact tri-state adjudication (all JSON types)
    if depth < MAX_DEPTH:
        if node.one_of:
            comp = _composition_verdict(node.one_of, value, path, depth,
                                        "oneOf")
            hits.extend(comp["hits"])
            notes.extend(comp["notes"])
            errors.extend(comp["errors"])
            if comp["state"] == FAIL:
                return {"state": FAIL, "hits": [], "notes": [],
                        "errors": comp["errors"]}
            if comp["state"] == PENDING:
                return pending()
        if node.any_of:
            comp = _composition_verdict(node.any_of, value, path, depth,
                                        "anyOf")
            hits.extend(comp["hits"])
            notes.extend(comp["notes"])
            errors.extend(comp["errors"])
            if comp["state"] == FAIL:
                return {"state": FAIL, "hits": [], "notes": [],
                        "errors": comp["errors"]}
            if comp["state"] == PENDING:
                return pending()

    if errors:
        return {"state": FAIL, "hits": [], "notes": [], "errors": errors}
    if hits or notes:
        return pending()
    return {"state": PASS, "hits": hits, "notes": notes, "errors": errors}


def _merge_child(sub: dict, hits: list, notes: list, errors: list) -> None:
    """Fold a child verdict (properties / items) into the parent buckets."""
    if sub["state"] == FAIL:
        errors.extend(sub["errors"])
        return
    hits.extend(sub["hits"])
    notes.extend(sub["notes"])


def _composition_verdict(branches: list, value: Any, path: str, depth: int,
                         keyword: str) -> dict:
    """Adjudicate one ``oneOf`` / ``anyOf`` site from per-branch states."""
    results = [coverage_verdict(b, value, path, depth + 1)
               for b in branches]
    n_pass = sum(1 for r in results if r["state"] == PASS)
    pending_results = [r for r in results if r["state"] == PENDING]
    fail_results = [r for r in results if r["state"] == FAIL]
    here = path or "#"

    hits: list = []
    notes: list = []
    errors: list[str] = []

    if keyword == "oneOf" and n_pass >= 2:
        errors.append(
            f"{here}: oneOf 要求恰好一个分支通过，但有 {n_pass} 个分支"
            "（按已知约束）确定通过")
        return {"state": FAIL, "hits": [], "notes": [], "errors": errors}

    if keyword == "anyOf" and n_pass >= 1:
        return {"state": PASS, "hits": [], "notes": [], "errors": errors}

    if keyword == "oneOf" and n_pass == 1 and not pending_results:
        return {"state": PASS, "hits": [], "notes": [], "errors": errors}

    # no clean pass settles the site: pending branches keep it undecided
    if pending_results:
        if keyword == "oneOf":
            if n_pass == 1:
                notes.append((
                    f"oneOf 已有一个分支确定通过，另有 {len(pending_results)} 个"
                    "分支仍可能通过，是否「恰好一个」无法判定", None))
            else:
                notes.append((
                    f"oneOf 无确定通过分支，{len(pending_results)} 个分支"
                    "仍可能通过，是否「恰好一个」无法判定", None))
        else:
            notes.append((
                f"anyOf 无确定通过分支，{len(pending_results)} 个分支"
                "仍可能通过，是否「至少一个」无法判定", None))
        for i, r in enumerate(results):
            if r["state"] != PENDING:
                continue
            hits.extend(r["hits"])
            for text, loc in r["notes"]:
                notes.append((f"分支 {i}: {text}", loc))
        return {"state": PENDING, "hits": hits, "notes": notes,
                "errors": errors}

    # every branch deterministically fails
    if keyword == "oneOf" and n_pass <= 1:
        errors.append(
            f"{here}: oneOf 的 {len(fail_results)} 个分支均确定不通过")
    else:
        errors.append(
            f"{here}: anyOf 的 {len(fail_results)} 个分支均确定不通过")
    for r in fail_results[:1]:
        errors.extend(r["errors"][:2])
    return {"state": FAIL, "hits": [], "notes": [], "errors": errors}


def minimal_instance(node: SchemaNode, depth: int = 0,
                     force_type: str | None = None) -> Any:
    """Smallest value that validates against ``node`` (best effort)."""
    if node.is_bool is True or node.is_cycle:
        return _scalar(force_type or "object", node, depth)
    if node.is_bool is False or node.invalid:
        return None
    # composition: prefer the smallest branch
    if (node.one_of or node.any_of) and depth < MAX_DEPTH:
        candidates = []
        for br in (node.one_of or node.any_of):
            v = minimal_instance(br, depth + 1)
            candidates.append(v)
        if candidates:
            return min(candidates, key=_value_rank)
    return _scalar(force_type or _primary_type(node), node, depth)


def _primary_type(node: SchemaNode) -> str:
    if node.const is not UNSET and node.const is not None:
        return _json_type(node.const)
    if node.enum:
        return _json_type(node.enum[0])
    if node.type:
        for t in node.type:
            if t != "null":
                return t
        return "null"
    if node.one_of or node.any_of:
        return "object"
    if node.properties or node.required or node.additional is False:
        return "object"
    if node.items is not None:
        return "array"
    return "string"


def _scalar(t: str, node: SchemaNode, depth: int) -> Any:
    if node.const is not UNSET and node.const is not None:
        return node.const
    if node.enum:
        return _smallest_enum(_enum_for_type(node.enum, t))
    if t == "null":
        return None
    if t == "boolean":
        return False
    if t in ("integer", "number"):
        return _smallest_number(node)
    if t == "string":
        return _smallest_string(node)
    if t == "array":
        return _smallest_array(node, depth)
    if t == "object":
        return _smallest_object(node, depth)
    return {}


def _smallest_enum(values: list) -> Any:
    vals = values or [None]
    def rank(v):
        t = _json_type(v)
        base = {"null": 0, "boolean": 1, "integer": 2, "number": 2,
                "string": 3, "array": 4, "object": 5}.get(t, 9)
        size = len(v) if isinstance(v, (list, dict, str)) else 0
        return (base, size, _json_key(v))
    return min(vals, key=rank)


def _enum_for_type(values: list, t: str) -> list:
    matching = [v for v in values if _json_type(v) == t
                or (t == "number" and _json_type(v) == "integer")]
    return matching or values


def _value_rank(v):
    t = _json_type(v)
    base = {"null": 0, "boolean": 1, "integer": 2, "number": 2,
            "string": 3, "array": 4, "object": 5}.get(t, 9)
    size = len(v) if isinstance(v, (list, dict, str)) else 0
    return (base, size)


def _smallest_number(node: SchemaNode):
    candidates = [0, 1, -1]
    if node.minimum is not None:
        candidates.append(node.minimum)
        candidates.append(node.minimum + 1)
    if node.ex_min is not None:
        candidates.append(node.ex_min + 1)
        if _is_int(node.ex_min):
            candidates.append(int(node.ex_min) + 1)
    if node.maximum is not None:
        candidates.append(node.maximum)
    if node.ex_max is not None:
        candidates.append(node.ex_max - 1)
    valid = [c for c in candidates if _number_fits(c, node)]
    if node.multiple_of:
        for c in list(valid):
            m = _round_to_multiple(c, node.multiple_of)
            if m is not None:
                valid.append(m)
        valid = [c for c in valid if _is_multiple(c, node.multiple_of)]
    if valid:
        return min(valid, key=lambda c: (abs(c), c))
    return 0


def _smallest_string(node: SchemaNode) -> str:
    length = node.min_length or 0
    s = "x" * length
    if node.max_length is not None and len(s) > node.max_length:
        s = ""
    return s


def _smallest_array(node: SchemaNode, depth: int) -> list:
    if depth >= MAX_DEPTH or node.items is None:
        count = node.min_items or 0
        return [_scalar("string", SchemaNode(loc=node.loc), depth + 1)
                for _ in range(min(count, 3))]
    count = node.min_items or 1
    if node.max_items is not None:
        count = min(count, node.max_items)
    count = min(count, 3)
    return [minimal_instance(node.items, depth + 1) for _ in range(count)]


def _smallest_object(node: SchemaNode, depth: int) -> dict:
    if depth >= MAX_DEPTH:
        return {}
    out: dict = {}
    # The canonical minimal instance contains ONLY required properties:
    # this keeps it valid on the favorable side and leaves optional fields
    # free to be injected by witness synthesis.
    names = sorted(node.required)
    if node.min_props and len(names) < node.min_props:
        existing = set(node.properties)
        extras: list[str] = []
        i = 0
        while len(names) + len(extras) < node.min_props:
            name = f"extra{i}"
            if name not in existing:
                extras.append(name)
            i += 1
        names += extras
    for name in names[:12]:
        child = node.properties.get(name)
        if child is None:
            if isinstance(node.additional, SchemaNode):
                out[name] = minimal_instance(node.additional, depth + 1)
            elif node.additional is not False:
                out[name] = ""
            continue
        if child.is_cycle:
            out[name] = {}  # break recursive expansion
        else:
            out[name] = minimal_instance(child, depth + 1)
    return out


def _number_fits(c, node: SchemaNode) -> bool:
    if node.minimum is not None and c < node.minimum:
        return False
    if node.maximum is not None and c > node.maximum:
        return False
    if node.ex_min is not None and c <= node.ex_min:
        return False
    if node.ex_max is not None and c >= node.ex_max:
        return False
    return True


def _is_multiple(value, multiple_of) -> bool:
    if value == 0:
        return True
    from fractions import Fraction
    try:
        q = Fraction(str(value)) / Fraction(str(multiple_of))
        return q.denominator == 1
    except (ValueError, ZeroDivisionError):
        return True


def _round_to_multiple(c, m):
    from fractions import Fraction
    try:
        fm = Fraction(str(m))
        q = Fraction(str(c)) / fm
        r = (q.numerator // q.denominator) * fm
        return float(r)
    except (ValueError, ZeroDivisionError):
        return None


def _is_int(v) -> bool:
    return float(v).is_integer()


def _esc(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _json_key(v: Any) -> str:
    import json
    return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)


def _json_safe_equal(a: Any, b: Any) -> bool:
    return _json_key(a) == _json_key(b)
