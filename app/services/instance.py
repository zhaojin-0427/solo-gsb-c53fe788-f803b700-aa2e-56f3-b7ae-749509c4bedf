"""Generate minimal valid instances and validate instances against nodes.

The generator produces the *smallest* value satisfying a normalized node
(fewer object keys, shorter strings/arrays), and follows reference cycles with
a depth budget so recursive schemas terminate. The validator is the same
constraint set the comparator reasons about; it is used to verify that a
generated counter-example passes exactly one side.
"""
from __future__ import annotations

from typing import Any

from .schemagraph import SchemaNode, UNSET

MAX_DEPTH = 8


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
    """Return a list of validation error messages (empty => valid)."""
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

    if vt == "array" and node.items is not None and depth < MAX_DEPTH:
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
        for i, item in enumerate(value):
            errors.extend(validate(node.items, item,
                                   f"{path}/{i}", depth + 1))

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
        for pname, pnode in node.properties.items():
            if pname in value and depth < MAX_DEPTH:
                # request/response filtering at replay time is done by the
                # caller; raw nodes keep both readOnly/writeOnly facets
                errors.extend(validate(pnode, value[pname],
                                       f"{path}/{_esc(pname)}", depth + 1))
        extra = {k: v for k, v in value.items() if k not in node.properties}
        if node.additional is False and extra:
            errors.append(
                f"{path or '#'}: 不允许额外属性 {sorted(extra)}")
        elif isinstance(node.additional, SchemaNode) and depth < MAX_DEPTH:
            for k, v in sorted(extra.items()):
                errors.extend(validate(node.additional, v,
                                       f"{path}/{_esc(k)}", depth + 1))

    # 组合分支按三态裁决（确定通过 / 确定失败 / 待定）：oneOf 要求
    # **恰好一个**分支确定通过——两个确定通过即校验失败；anyOf 要求
    # **至少一个**。存在待定分支时此处不下定论（校验既未确定失败），
    # 由 unsupported_along_value 沿值上报待定关键字与位置。
    if node.one_of and depth < MAX_DEPTH:
        verdicts = _branch_verdicts(node.one_of, value, path, depth)
        n_pass = verdicts.count(_BRANCH_PASS)
        if n_pass >= 2:
            errors.append(
                f"{path or '#'}: 同时满足 oneOf 的 {n_pass} 个分支"
                "（要求恰好一个）")
        elif n_pass == 0 and _BRANCH_UNDECIDED not in verdicts:
            errors.append(f"{path or '#'}: 不满足 oneOf 的任何分支")
    if node.any_of and depth < MAX_DEPTH:
        verdicts = _branch_verdicts(node.any_of, value, path, depth)
        if _BRANCH_PASS not in verdicts and \
                _BRANCH_UNDECIDED not in verdicts:
            errors.append(f"{path or '#'}: 不满足 anyOf 的任何分支")
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


def unsupported_along_value(node: SchemaNode, value: Any,
                            depth: int = 0) -> list:
    """Unsupported keywords whose skipped constraints decide ``value``'s fate.

    Walks the parsed schema graph *along the actual value* — local ``$ref``
    (already resolved into the graph), nested properties, array items and
    ``oneOf``/``anyOf`` branches — and returns ``(keyword, FacetLoc)`` for
    every unsupported keyword the value's verdict still depends on. Optional
    properties absent from the value are not visited, and keywords that do
    not apply to the value's JSON type (e.g. ``pattern`` for an integer) are
    ignored. Compositions are adjudicated per branch state
    （确定通过 / 确定失败 / 待定）: for ``oneOf`` a single cleanly passing
    branch only settles the verdict when every other branch definitely
    fails — a sibling that may still pass (``pattern`` …) keeps the row
    undetermined; for ``anyOf`` one cleanly passing branch is enough.
    """
    hits: list = []
    if node is None or depth >= MAX_DEPTH:
        return hits
    if node.is_bool is not None or node.is_cycle or node.invalid:
        return hits  # bool/cycle/invalid nodes are reported by the caller

    vt = _json_type(value)
    for kw, loc in node.unsupported:
        applies = _UNSUPPORTED_KW_TYPES.get(kw)
        if applies is None or vt in applies:
            hits.append((kw, loc))

    if vt == "object":
        for pname, pnode in node.properties.items():
            if pname in value:
                hits.extend(unsupported_along_value(
                    pnode, value[pname], depth + 1))
        if isinstance(node.additional, SchemaNode):
            extra = [k for k in value if k not in node.properties]
            for k in sorted(extra):
                hits.extend(unsupported_along_value(
                    node.additional, value[k], depth + 1))
    elif vt == "array" and node.items is not None:
        for item in value:
            hits.extend(unsupported_along_value(node.items, item, depth + 1))

    # oneOf「恰好一个」：一个分支确定通过时，任何仍可能通过的待定分支
    # 都让整体无法判定（它若也通过就违反恰好一个）；没有确定通过时，
    # 整体能否通过同样取决于待定分支。两个确定通过属于校验失败，
    # 由 validate 报告，这里不再重复列举。
    if node.one_of and depth < MAX_DEPTH:
        verdicts = _branch_verdicts(node.one_of, value, "", depth)
        if verdicts.count(_BRANCH_PASS) <= 1:
            for branch, verdict in zip(node.one_of, verdicts):
                if verdict == _BRANCH_UNDECIDED:
                    hits.extend(_pending_hits(branch, value, depth + 1))
    # anyOf「至少一个」：已有一个确定通过即覆盖（其余分支待定不影响
    # 结论）；否则整体能否通过取决于待定分支。
    if node.any_of and depth < MAX_DEPTH:
        verdicts = _branch_verdicts(node.any_of, value, "", depth)
        if _BRANCH_PASS not in verdicts:
            for branch, verdict in zip(node.any_of, verdicts):
                if verdict == _BRANCH_UNDECIDED:
                    hits.extend(_pending_hits(branch, value, depth + 1))
    return hits


# ============================================================ branch verdicts
# 组合分支（oneOf / anyOf）相对某个实际取值的三态裁决。
_BRANCH_PASS = "pass"              # 确定通过：已知约束全过且无待定因素
_BRANCH_FAIL = "fail"              # 确定失败：已支持约束已经失败
_BRANCH_UNDECIDED = "undecided"    # 待定：通过与否仍依赖未支持/不可判定约束


def _branch_verdicts(branches, value, path: str, depth: int) -> list:
    """Classify each combinator branch for ``value`` into the three states.

    * 确定失败：已支持（可判定）约束已经失败——未支持或不可判定的约束
      只会更严格，不可能让分支重新通过；
    * 待定：已知约束全过，但通过与否仍依赖 ``pattern`` 等未支持关键字，
      或取值触达了循环引用 / 无法解析的节点；
    * 确定通过：已知约束全过且没有任何待定因素。
    """
    verdicts: list = []
    for b in branches:
        if _undecidable_along_value(b, value, depth + 1):
            verdicts.append(_BRANCH_UNDECIDED)
        elif validate(b, value, path, depth + 1):
            verdicts.append(_BRANCH_FAIL)
        elif unsupported_along_value(b, value, depth + 1):
            verdicts.append(_BRANCH_UNDECIDED)
        else:
            verdicts.append(_BRANCH_PASS)
    return verdicts


def _pending_hits(node: SchemaNode, value: Any, depth: int) -> list:
    """待定分支需要列出的原因：未支持关键字，以及循环/不可判定节点位置。"""
    hits = unsupported_along_value(node, value, depth)
    hits.extend(_undecidable_along_value(node, value, depth))
    return hits


def _undecidable_along_value(node: SchemaNode, value: Any,
                             depth: int = 0) -> list:
    """Nodes along ``value`` whose constraints cannot be decided at all.

    :func:`unsupported_along_value` reports unsupported *keywords*; this
    walk covers the harder undecidable cases — reference cycles and
    invalid / unresolvable schemas — so that a combinator branch touching
    them is treated as 待定 instead of being optimistically counted as
    确定通过. Untouched optional properties are not visited.
    """
    hits: list = []
    if node is None or depth >= MAX_DEPTH:
        return hits
    if node.is_cycle:
        hits.append(("循环引用", node.loc))
        return hits
    if node.invalid:
        hits.append(("模式不可判定", node.loc))
        return hits
    if node.is_bool is not None:
        return hits
    vt = _json_type(value)
    if vt == "object":
        for pname, pnode in node.properties.items():
            if pname in value:
                hits.extend(_undecidable_along_value(
                    pnode, value[pname], depth + 1))
        if isinstance(node.additional, SchemaNode):
            for k in sorted(k for k in value if k not in node.properties):
                hits.extend(_undecidable_along_value(
                    node.additional, value[k], depth + 1))
    elif vt == "array" and node.items is not None:
        for item in value:
            hits.extend(_undecidable_along_value(node.items, item,
                                                 depth + 1))
    for branches in (node.one_of, node.any_of):
        for b in branches:
            hits.extend(_undecidable_along_value(b, value, depth + 1))
    return hits


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
