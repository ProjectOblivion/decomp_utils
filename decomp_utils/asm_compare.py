import re
import codecs
from collections import namedtuple
from types import SimpleNamespace
from pathlib import Path
from collections import defaultdict, deque
from hashlib import sha1
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from .helpers import RE_PATTERNS
import Levenshtein

__all__ = [
    "group_by_hash",
    "get_buckets",
    "compare",
    "group_results",
    "build_hierarchy",
    "parse_files",
    "find_matches",
    "generate_clusters",
    "best_results",
    "Result",
]

AsmLine = namedtuple("AsmLine", ["offset", "address", "word", "op", "fields"])
JtblLine = namedtuple(
    "JtblLine", ["offset", "address", "jump_address", "data_size", "jump_label"]
)
ParsedAsmFile = namedtuple(
    "ParsedAsmFile", ["path", "hash", "ops", "instructions", "jtbls"]
)
ParsedOps = namedtuple("ParsedOps", ["path", "hash", "parsed"])
ParsedInstructions = namedtuple(
    "ParsedInstructions", ["path", "hash", "words", "parsed", "normalized"]
)
ParsedJtbl = namedtuple("ParsedJtbl", ["name", "lines"])
Result = namedtuple("Result", ["ref_op_hash", "check_op_hash", "score", "debug"])
ResultDebug = namedtuple(
    "ResultDebug",
    [
        "string_similarity",
        "sequence_similarity",
        "weighted_score",
        "composite_score",
        "dynamic_threshold",
        "mismatches",
    ],
)


# class SotnClusterMap:
# Todo: Convert this to a class
def generate_clusters(version, overlays, threshold=0.95, exclude=[], debug=False):
    """Generate a report of duplicate functions."""

    def add_family(parent, children, indent=0, debug=False):
        """Recursively print a family of functions."""
        paths = (func.path for func in funcs_by_op_hash[parent])
        for i, path in enumerate(sorted(paths, key=lambda x: str(x))):
            if debug and (debug_obj := children.get("debug", None)):
                debug_string = (
                    f"path={path}, "
                    f"string_similarity={debug_obj.string_similarity}, "
                    f"sequence_similarity={debug_obj.sequence_similarity}, "
                    f"weighted_score={debug_obj.weighted_score}, "
                    f"composite_score={debug_obj.composite_score}, "
                    f"dynamic_threshold={debug_obj.dynamic_threshold}, "
                    f"mismatches={debug_obj.mismatches}"
                )
                print(debug_string)

            item = SimpleNamespace(
                score=children.get("score") or 1.00,
                decomp_status="matchings" in path.parts,
                name=path.stem,
                path=path,
                indent=indent,
            )
            cluster.append(item)

        # Proceed to next generation
        for parent, children in (
            (p, c) for child in children.get("children") or [] for p, c in child.items()
        ):
            add_family(parent, children, (indent or 1) + 2)

    files = (
        dirpath / f
        for dirpath, _, filenames in Path("asm").joinpath(version).walk()
        if "data" not in dirpath.parts
        and (
            "all" in overlays
            or any(x in dirpath.parts or f"{x}_psp" in dirpath.parts for x in overlays)
        )
        and not any(x in dirpath.parts for x in exclude)
        for f in filenames
    )
    funcs_by_op_hash = group_by_hash(
        (func.ops.hash, func) for func in parse_files(files)
    )
    ops_by_op_hash = {k: v[0].ops.parsed for k, v in funcs_by_op_hash.items()}
    buckets = get_buckets((ops_by_op_hash,), tolerance=0.1, num_buckets=25)

    kwargs = (
        {"ops_by_op_hash": bucket[0], "threshold": threshold} for bucket in buckets
    )
    with ProcessPoolExecutor() as executor:
        results = executor.map(find_matches, kwargs)

    hierarchy_group = build_hierarchy(
        [k for k, v in funcs_by_op_hash.items() if len(v) > 1],
        group_results(results),
    )

    clusters = []
    flattened_hierarchy = (
        (p, c) for hierarchy in hierarchy_group for p, c in hierarchy.items()
    )
    for parent, children in flattened_hierarchy:
        cluster = []
        add_family(parent, children, debug=debug)
        clusters.append(cluster)

    return clusters


def best_results(results):
    results_by_hash = defaultdict(list)
    for result in results:
        results_by_hash[result.check_op_hash].append(result)
    sorted_score_groups = (
        sorted(group, key=lambda x: x.score, reverse=True)
        for group in results_by_hash.values()
    )
    return (
        result
        for group in sorted_score_groups
        for result in group
        if result.score == group[0].score
    )


def parse_file(path, parse_instructions=True, parse_jtbls=False):
    parsed_ops, parsed_instructions, parsed_jtbls = None, None, None
    file_bytes = path.read_bytes()
    # file_hash = sha1(file_bytes).hexdigest()
    file_hash = None
    rodata_section_start = file_bytes.find(b".section .rodata")
    text_section_start = (
        file_bytes.find(b".section .text")
        if b".section .text" in file_bytes
        else file_bytes.find(b"glabel " + path.stem.encode())
    )
    if text_section_start != -1:
        # Todo: add in ending index if rodata start > text start
        text_slice = file_bytes[text_section_start:]
        if not parse_instructions and (
            ops := tuple(
                op.decode("utf-8") for op in RE_PATTERNS.op.findall(text_slice)
            )
        ):
            # Todo: See if using hash() is faster
            parsed_ops = ParsedOps(path, sha1("".join(ops).encode()).hexdigest(), ops)
        else:
            asm_lines = tuple(
                AsmLine(
                    codecs.decode(line_match.group("offset").ljust(8, b"0"), "hex"),
                    codecs.decode(line_match.group("address").ljust(8, b"0"), "hex"),
                    codecs.decode(line_match.group("word").ljust(8, b"0"), "hex"),
                    line_match.group("op").decode("utf-8"),
                    line_match.group("fields").decode("utf-8"),
                )
                for line_match in RE_PATTERNS.asm_line.finditer(text_slice)
            )
            if asm_lines:
                ops = tuple(asm_line.op for asm_line in asm_lines)
                if ops:
                    parsed_ops = ParsedOps(
                        path, sha1("".join(ops).encode()).hexdigest(), ops
                    )
                else:
                    parsed_ops = None
                normalized_instructions = tuple(
                    f'{asm_line.op} {RE_PATTERNS.masking.sub("", asm_line.fields)}'
                    for asm_line in asm_lines
                )
                words = tuple(asm_line.word for asm_line in asm_lines)
                instructions = tuple(
                    f"{asm_line.op} {asm_line.fields}" for asm_line in asm_lines
                )
                # Todo: See if using hash() is faster
                parsed_instructions = ParsedInstructions(
                    path,
                    sha1("".join(normalized_instructions).encode()).hexdigest(),
                    words,
                    instructions,
                    normalized_instructions,
                )

    if parse_jtbls and rodata_section_start != -1:
        rodata_slice = file_bytes[rodata_section_start:text_section_start]
        parsed_jtbls = (
            ParsedJtbl(
                match.group("name"),
                tuple(
                    JtblLine(
                        codecs.decode(line_match.group("offset").ljust(8, b"0"), "hex"),
                        codecs.decode(
                            line_match.group("address").ljust(8, b"0"), "hex"
                        ),
                        codecs.decode(line_match.group("data").ljust(8, b"0"), "hex"),
                        line_match.group("data_type").decode("utf-8"),
                        line_match.group("location").decode("utf-8"),
                    )
                    for line_match in RE_PATTERNS.jtbl_line.finditer(
                        match.group("table")
                    )
                ),
            )
            for match in RE_PATTERNS.jtbl.finditer(rodata_slice)
        )

    return ParsedAsmFile(path, file_hash, parsed_ops, parsed_instructions, parsed_jtbls)


def parse_files(files):
    with ThreadPoolExecutor() as executor:
        parsed_files = tuple(
            parsed_asm
            for parsed_asm in executor.map(parse_file, files)
            if parsed_asm.ops
        )
    return parsed_files


def group_by_hash(funcs, hash_type=""):
    by_hash = defaultdict(list)
    if hash_type.startswith("op"):
        funcs = ((func.ops.hash, func) for func in funcs)
    elif hash_type.startswith("instruction"):
        funcs = ((func.instructions.hash, func) for func in funcs)

    for hash, func in funcs:
        by_hash[hash].append(func)
    return by_hash


def compare(ref_ops, check_ops, threshold, debug=False):
    # Evaluate similarity on a per character basis
    string_similarity = Levenshtein.ratio(
        ref_ops, check_ops, processor=lambda x: " ".join(x)
    )
    if string_similarity >= threshold - 0.1:
        sequence_similarity = calculate_sequence_similarity(ref_ops, check_ops)
        score, weighted_score, composite_score = calculate_score(
            string_similarity, sequence_similarity
        )

        # Allow for a lower threshold based on the length difference of the two ops lists
        length_factor = min(len(ref_ops), len(check_ops)) / max(
            len(ref_ops), len(check_ops)
        )

        # Use 75% of the base threshold with the remaining 25% being determined dynamically, but not less than 5% less than the base threshold
        dynamic_threshold = max(
            threshold - 0.05, threshold * (0.75 + 0.25 * length_factor)
        )

        mismatches = tuple(
            (i, ref_op, check_op)
            for i, (ref_op, check_op) in enumerate(zip(ref_ops, check_ops))
            if ref_op != check_op
        )

        if debug:
            return (
                score,
                dynamic_threshold,
                ResultDebug(
                    string_similarity,
                    sequence_similarity,
                    weighted_score,
                    composite_score,
                    dynamic_threshold,
                    mismatches,
                ),
            )
        else:
            return score, dynamic_threshold, None
    else:
        return None, None, None


def calculate_score(string_similarity, sequence_similarity):
    # Evaluate confidence, with closer string and sequence similarities giving higher confidence
    confidence = 1 - abs(string_similarity - sequence_similarity)

    weighted_score = min(
        max(string_similarity, sequence_similarity),
        (0.45 * max(string_similarity, sequence_similarity))
        + (0.35 * min(string_similarity, sequence_similarity))
        + (0.2 * confidence),
    )

    # Don't let confidence raise the score above the higher of the two ratios
    composite_score = min(
        max(string_similarity, sequence_similarity),
        (string_similarity + sequence_similarity + confidence) / 3,
    )

    # Unsure which is these gives more accurate scoring, so using both for now and taking the higher one
    return max(weighted_score, composite_score), weighted_score, composite_score


def find_matches(kwargs, debug=True):
    results = []
    if "ops_by_op_hash" in kwargs:
        queue = deque(kwargs["ops_by_op_hash"].items())
        while queue:
            ref_ops_hash, ref_ops = queue.popleft()
            for check_ops_hash, check_ops in queue:
                if ref_ops_hash == check_ops_hash:
                    score, dynamic_threshold, result_debug = 1.0, 0.95, None
                else:
                    score, dynamic_threshold, result_debug = compare(
                        ref_ops, check_ops, kwargs["threshold"], debug=debug
                    )
                if score and dynamic_threshold and score >= dynamic_threshold:
                    results.append(
                        Result(ref_ops_hash, check_ops_hash, score, result_debug)
                    )
    elif "ref_ops_by_op_hash" in kwargs and "check_ops_by_op_hash" in kwargs:
        for ref_ops_hash, ref_ops in kwargs["ref_ops_by_op_hash"].items():
            for check_ops_hash, check_ops in kwargs["check_ops_by_op_hash"].items():
                if ref_ops_hash == check_ops_hash:
                    score, dynamic_threshold, result_debug = 1.0, 0.95, None
                else:
                    score, dynamic_threshold, result_debug = compare(
                        ref_ops, check_ops, kwargs["threshold"], debug=debug
                    )
                if score and dynamic_threshold and score >= dynamic_threshold:
                    results.append(
                        Result(ref_ops_hash, check_ops_hash, score, result_debug)
                    )
    return tuple(results)


def calculate_sequence_similarity(ref_ops, check_ops):
    """Calculate positional similarity by checking if instructions exist within a small window."""
    window = max(1, min(len(ref_ops), len(check_ops)) // 10)
    missed_ops = 0
    for i, ref_op in enumerate(ref_ops):
        start = max(0, i - window)
        end = min(len(check_ops), i + window + 1)
        missed_ops += 1 if ref_op not in check_ops[start:end] else 0
    avg_len = (len(ref_ops) + len(ref_ops)) / 2
    positional_similarity = 1 - (missed_ops / avg_len)

    # Evaluate similarity per "word" instead of per character
    list_similarity = Levenshtein.seqratio(ref_ops, check_ops)

    # list and positional similarities evaluate the same metric in different ways, so we take the higher one
    return max(list_similarity, positional_similarity)


### Experimental algorithms, currently unused due to computation cost ###
def dtw_distance(ref_ops, check_ops):
    import numpy as np

    m, n = len(ref_ops), len(check_ops)
    dtw = np.full((m + 1, n + 1), float("inf"))
    dtw[0][0] = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_ops[i - 1] == check_ops[j - 1] else 1
            dtw[i][j] = cost + min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1])
    return dtw[m][n]


def dtw_similarity(ref_ops, check_ops):
    distance = dtw_distance(ref_ops, check_ops)
    max_len = max(len(ref_ops), len(check_ops))
    return 1 - (distance / max_len) if max_len != 0 else 0


### End experimental ###


def get_buckets(groups_by_hash, tolerance=0.1, num_buckets=20):
    sizes = {len(members) for group in groups_by_hash for members in group.values()}
    min_size = min(sizes)
    max_size = max(sizes)
    bucket_size = (max_size - min_size) / num_buckets

    buckets = []
    for bucket_index in range(num_buckets):
        bucket_min = (min_size + bucket_index * bucket_size) * (1 - tolerance)
        bucket_max = (bucket_min + bucket_size) * (1 + tolerance)
        bucket = tuple(
            {k: v for k, v in group.items() if bucket_min <= len(v) <= bucket_max}
            for group in groups_by_hash
        )
        # Todo: Should this check all elements in bucket?
        if bucket[0]:
            buckets.append(bucket)

    return buckets


def group_results(results):
    flattened_results = {element for sublist in results for element in sublist}
    results = tuple(best_results(flattened_results))
    result_map = defaultdict(list)
    for result in results:
        result_map[result.ref_op_hash].append(result)
        result_map[result.check_op_hash].append(result)

    visited = set()
    groups = defaultdict(list)
    for result in results:
        if result not in visited:
            # deque creates a list from the outer iterable, so nesting result treats an iterable as the queue item.
            queue = deque([result])
            while queue:
                current = queue.popleft()
                if current not in visited:
                    visited.add(current)
                    groups[current.ref_op_hash].append(
                        (current.check_op_hash, current.score, current.debug)
                    )
                    queue.extend(
                        result_map[current.ref_op_hash]
                        + result_map[current.check_op_hash]
                    )
    return {k: sorted(v, key=lambda x: x[1], reverse=True) for k, v in groups.items()}


def get_children(score_map, ratio_map, child_map, parent):
    child = child_map.get(parent)
    if child:
        return {
            parent: {
                "children": [
                    get_children(score_map, ratio_map, child_map, c) for c in child
                ],
                "score": score_map.get(parent),
                "debug": ratio_map.get(parent),
            }
        }
    else:
        return {
            parent: {
                "children": None,
                "score": score_map.get(parent),
                "debug": ratio_map.get(parent),
            }
        }


def build_hierarchy(full_matches, grouped_results):
    child_map = defaultdict(list)
    score_map = {}
    debug_map = {}
    for ref_op_hash, results in grouped_results.items():
        for check_op_hash, score, debug in results:
            child_map[ref_op_hash].append(check_op_hash)
            score_map[check_op_hash] = score
            debug_map[check_op_hash] = debug
    roots = [
        parent
        for parent in child_map
        if parent
        not in [child for children in child_map.values() for child in children]
    ]
    hierarchy = []

    for ref_op_hash, results in grouped_results.items():
        if ref_op_hash in roots:
            hierarchy.append(get_children(score_map, debug_map, child_map, ref_op_hash))

    hierarchy.extend(
        {op_hash: {"children": None, "score": 1.00, "debug": None}}
        for op_hash in full_matches
        if op_hash not in child_map and op_hash not in score_map
    )
    return hierarchy
