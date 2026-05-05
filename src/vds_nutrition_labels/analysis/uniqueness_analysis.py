#!/usr/bin/env python3
"""
Importable near-duplicate / uniqueness analysis for mixed-language source folders.

Supported by default:
  C:      .c, .h
  C++:    .cpp, .cc, .cxx, .c++, .hpp, .hh, .hxx, .ipp, .tpp
  Java:   .java
  Python: .py, .pyw

Public usage:

    from uniqueness_runner import run_uniqueness_detection, UniquenessConfig

    result = run_uniqueness_detection(
        "/path/to/source/folder",
        UniquenessConfig(
            output_dir="uniqueness_results",
            normalize_literals=True,
            normalize_identifiers=True,
        ),
    )

    print(result.metrics)
    print(result.clusters_path)

This module creates the compressed JSONL token input expected by Microsoft's
near-duplicate-code-detector, runs the detector, and writes summary metrics.

External requirements when tokens_only=False:
  - git
  - dotnet SDK
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import keyword
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize as py_tokenize
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

REPO_URL = "https://github.com/Microsoft/near-duplicate-code-detector.git"
REPO_NAME = "near-duplicate-code-detector"

LANGUAGE_EXTENSIONS: dict[str, list[str]] = {
    "c": [".c", ".h"],
    "cpp": [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".ipp", ".tpp"],
    "java": [".java"],
    "python": [".py", ".pyw"],
}

GENERIC_EXTENSIONS = [
    ".js", ".jsx", ".ts", ".tsx", ".cs", ".go", ".rs", ".php", ".rb",
    ".kt", ".kts", ".scala", ".swift",
]

DEFAULT_LANGUAGES = ["c", "cpp", "java", "python"]
EXTENSION_LANGUAGE = {ext: lang for lang, exts in LANGUAGE_EXTENSIONS.items() for ext in exts}
DEFAULT_EXTENSIONS = sorted({ext for lang in DEFAULT_LANGUAGES for ext in LANGUAGE_EXTENSIONS[lang]})

DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".venv", "venv", "env", "node_modules", "dist", "build", "target", "out",
    ".idea", ".vscode", "cmake-build-debug", "cmake-build-release",
}

PY_SKIP_TOKEN_TYPES = {
    py_tokenize.ENCODING,
    py_tokenize.NL,
    py_tokenize.NEWLINE,
    py_tokenize.INDENT,
    py_tokenize.DEDENT,
    py_tokenize.COMMENT,
    py_tokenize.ENDMARKER,
}

C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else",
    "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long", "register",
    "restrict", "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef",
    "union", "unsigned", "void", "volatile", "while", "_Alignas", "_Alignof", "_Atomic",
    "_Bool", "_Complex", "_Generic", "_Imaginary", "_Noreturn", "_Static_assert", "_Thread_local",
    "true", "false", "nullptr", "NULL",
}

CPP_KEYWORDS = C_KEYWORDS | {
    "alignas", "alignof", "and", "and_eq", "asm", "bitand", "bitor", "bool", "catch", "char8_t",
    "char16_t", "char32_t", "class", "compl", "concept", "consteval", "constexpr", "constinit",
    "const_cast", "co_await", "co_return", "co_yield", "decltype", "delete", "dynamic_cast",
    "explicit", "export", "friend", "mutable", "namespace", "new", "noexcept", "not", "not_eq",
    "operator", "or", "or_eq", "private", "protected", "public", "reinterpret_cast", "requires",
    "static_assert", "static_cast", "template", "this", "thread_local", "throw", "try", "typeid",
    "typename", "using", "virtual", "wchar_t", "xor", "xor_eq", "override", "final", "import", "module",
}

JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const",
    "continue", "default", "do", "double", "else", "enum", "exports", "extends", "final", "finally",
    "float", "for", "goto", "if", "implements", "import", "instanceof", "int", "interface", "long",
    "module", "native", "new", "open", "opens", "package", "permits", "private", "protected", "provides",
    "public", "record", "requires", "return", "sealed", "short", "static", "strictfp", "super", "switch",
    "synchronized", "this", "throw", "throws", "to", "transient", "transitive", "try", "uses", "var",
    "void", "volatile", "while", "with", "yield", "true", "false", "null",
}

LANGUAGE_KEYWORDS = {
    "python": set(keyword.kwlist) | {"True", "False", "None"},
    "c": C_KEYWORDS,
    "cpp": CPP_KEYWORDS,
    "java": JAVA_KEYWORDS,
    "generic": set(),
}

IDENTIFIER_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
NUMBER_RE = re.compile(
    r"0[xX][0-9A-Fa-f][0-9A-Fa-f_']*"
    r"|0[bB][01][01_']*"
    r"|(?:\d[\d_']*\.\d[\d_']*|\.\d[\d_']*|\d[\d_']*\.)(?:[eEpP][+-]?\d[\d_']*)?"
    r"|\d[\d_']*(?:[eEpP][+-]?\d[\d_']*)?(?:[uUlLfF]*)"
)

OPERATORS = sorted(
    [
        ">>>=", "<<=", ">>=", "...", "->*", ".*", "<=>", "::", "++", "--", "->", "==", "!=", "<=", ">=",
        "&&", "||", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<", ">>", "##", "=>",
        "@", "#", "{", "}", "[", "]", "(", ")", ";", ",", ".", ":", "?", "~", "!", "+", "-", "*", "/",
        "%", "&", "|", "^", "<", ">", "=",
    ],
    key=len,
    reverse=True,
)

GENERIC_TOKEN_RE = re.compile(
    r'''
    [A-Za-z_$][A-Za-z0-9_$]*
    | 0[xX][0-9A-Fa-f][0-9A-Fa-f_']*
    | 0[bB][01][01_']*
    | \d+\.\d+(?:[eE][+-]?\d+)?
    | \d+(?:[eE][+-]?\d+)?
    | "(?:\\.|[^"\\])*"
    | '(?:\\.|[^'\\])*'
    | `(?:\\.|[^`\\])*`
    | >>>=|<<=|>>=|===|!==|==|!=|<=|>=|=>|&&|\|\||->|::|\+\+|--|<<|>>
    | [][{}();,.:+\-*/%&|^!~<>?=#@]
    ''',
    re.VERBOSE | re.DOTALL,
)


@dataclass
class FileRecord:
    relative_path: str
    absolute_path: str
    language: str
    size_bytes: int
    sha256: str
    token_count: int
    status: str
    note: str = ""


@dataclass
class UniquenessConfig:
    output_dir: Path | str = Path("uniqueness_results")

    languages: Sequence[str] = field(default_factory=lambda: tuple(DEFAULT_LANGUAGES))
    extensions: Sequence[str] | None = None

    exclude_dirs: Sequence[str] = field(default_factory=tuple)
    include_hidden: bool = False
    follow_symlinks: bool = False
    max_file_size_mb: float = 5.0
    min_tokens: int = 5

    normalize_literals: bool = False
    normalize_identifiers: bool = False

    tool_root: Path | str | None = None
    patch_detector: bool = True
    build_detector: bool = True
    target_framework: str = "net8.0"
    detector_min_tokens: int = 5

    key_jaccard_threshold: float = 0.8
    jaccard_threshold: float = 0.7
    output_prefix: str = "duplicate_clusters"

    tokens_only: bool = False
    keep_temp: bool = False
    quiet: bool = False


@dataclass
class UniquenessResult:
    output_dir: Path
    token_file: Path
    manifest_csv: Path
    language_summary_csv: Path

    records: list[FileRecord]
    included_count: int
    skipped_or_errored_count: int

    clusters_path: Path | None = None
    detector_log: Path | None = None
    duplicate_pairs_csv: Path | None = None
    metrics_path: Path | None = None
    metrics: dict | None = None


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    if not quiet:
        print("$", " ".join(map(str, cmd)))

    return subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required executable '{name}' was not found on PATH. "
            "Install it first, or use tokens_only=True to only create detector input."
        )


def resolve_detector_project(tool_root: Path) -> Path | None:
    candidates = [
        tool_root / "DuplicateCodeDetector" / "DuplicateCodeDetector.csproj",
        tool_root / "DuplicateCodeDetector.csproj",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None


def patch_detector_sources(
    csproj: Path,
    min_tokens_for_detector: int,
    target_framework: str,
) -> None:
    project_dir = csproj.parent

    csproj_text = csproj.read_text(encoding="utf-8-sig")
    csproj_text = re.sub(
        r"<TargetFramework>[^<]+</TargetFramework>",
        f"<TargetFramework>{target_framework}</TargetFramework>",
        csproj_text,
    )
    csproj_text = re.sub(
        r'(<PackageReference\s+Include="Newtonsoft\.Json"\s+Version=")[^"]+("\s*/>)',
        r"\g<1>13.0.3\2",
        csproj_text,
    )
    csproj.write_text(csproj_text, encoding="utf-8")

    clone_detector = project_dir / "CloneDetector.cs"
    if clone_detector.exists():
        clone_text = clone_detector.read_text(encoding="utf-8-sig")
        clone_text_new = re.sub(
            r"private\s+const\s+int\s+MIN_NUM_TOKENS_FOR_FILE\s*=\s*\d+\s*;",
            f"private const int MIN_NUM_TOKENS_FOR_FILE = {min_tokens_for_detector};",
            clone_text,
        )
        if clone_text_new != clone_text:
            clone_detector.write_text(clone_text_new, encoding="utf-8")


def ensure_detector(
    output_dir: Path,
    tool_root_arg: Path | str | None,
    *,
    patch: bool,
    build: bool,
    min_tokens_for_detector: int,
    target_framework: str,
    quiet: bool,
) -> Path:
    require_executable("dotnet")

    tool_root = (
        Path(tool_root_arg).expanduser().resolve()
        if tool_root_arg
        else (output_dir / "external" / REPO_NAME).resolve()
    )

    csproj = resolve_detector_project(tool_root)

    if csproj is None:
        require_executable("git")
        tool_root.parent.mkdir(parents=True, exist_ok=True)

        if not quiet:
            print(f"Detector not found. Cloning to: {tool_root}")

        run_command(["git", "clone", REPO_URL, str(tool_root)], quiet=quiet)
        csproj = resolve_detector_project(tool_root)

        if csproj is None:
            raise RuntimeError(f"Could not find DuplicateCodeDetector.csproj below {tool_root}")

    if patch:
        patch_detector_sources(csproj, min_tokens_for_detector, target_framework)

    if build:
        if not quiet:
            print("Building detector...")

        result = run_command(
            ["dotnet", "build", str(csproj), "--configuration", "Release"],
            quiet=quiet,
        )

        if not quiet and result.stdout.strip():
            print(result.stdout.strip())
        if not quiet and result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)

    return csproj


def language_from_path(path: Path) -> str:
    return EXTENSION_LANGUAGE.get(path.suffix.lower(), "generic")


def extensions_for_languages(languages: Sequence[str]) -> set[str]:
    extensions: set[str] = set()

    for language in languages:
        extensions.update(GENERIC_EXTENSIONS if language == "generic" else LANGUAGE_EXTENSIONS[language])

    return extensions


def should_skip_dir(
    path: Path,
    *,
    include_hidden: bool,
    exclude_dirs: set[str],
) -> bool:
    return path.name in exclude_dirs or (not include_hidden and path.name.startswith("."))


def iter_source_files(
    input_dir: Path,
    output_dir: Path,
    extensions: set[str],
    *,
    include_hidden: bool,
    follow_symlinks: bool,
    exclude_dirs: set[str],
    max_file_size_bytes: int,
) -> Iterator[Path]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    for root, dirs, files in os.walk(input_dir, followlinks=follow_symlinks):
        root_path = Path(root)

        dirs[:] = [
            d
            for d in dirs
            if not should_skip_dir(
                root_path / d,
                include_hidden=include_hidden,
                exclude_dirs=exclude_dirs,
            )
        ]

        for filename in files:
            path = root_path / filename

            if not include_hidden and any(part.startswith(".") for part in path.relative_to(input_dir).parts):
                continue

            if path.suffix.lower() not in extensions:
                continue

            try:
                resolved = path.resolve()

                if resolved == output_dir or output_dir in resolved.parents:
                    continue

                if resolved.is_symlink() and not follow_symlinks:
                    continue

                if resolved.stat().st_size > max_file_size_bytes:
                    continue

            except OSError:
                continue

            yield path


def read_text_lossy(path: Path) -> str:
    data = path.read_bytes()

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def normalize_python_token(
    tok_type: int,
    tok: str,
    *,
    normalize_literals: bool,
    normalize_identifiers: bool,
) -> str:
    if normalize_identifiers and tok_type == py_tokenize.NAME and tok not in LANGUAGE_KEYWORDS["python"]:
        return "<ID>"

    if normalize_literals and tok_type == py_tokenize.NUMBER:
        return "<NUM>"

    if normalize_literals and tok_type == py_tokenize.STRING:
        return "<STR>"

    return tok


def tokenize_python(
    text: str,
    *,
    normalize_literals: bool,
    normalize_identifiers: bool,
) -> list[str]:
    tokens: list[str] = []

    try:
        for tok in py_tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in PY_SKIP_TOKEN_TYPES:
                continue

            tokens.append(
                normalize_python_token(
                    tok.type,
                    tok.string,
                    normalize_literals=normalize_literals,
                    normalize_identifiers=normalize_identifiers,
                )
            )

    except (py_tokenize.TokenError, IndentationError):
        return tokenize_generic(
            text,
            normalize_literals=normalize_literals,
            normalize_identifiers=normalize_identifiers,
        )

    return tokens


def consume_quoted_literal(text: str, i: int, quote: str) -> tuple[str, int]:
    j = i + 1

    while j < len(text):
        if text[j] == "\\":
            j += 2
            continue

        if text[j] == quote:
            j += 1
            break

        j += 1

    return text[i:j], j


def consume_java_text_block(text: str, i: int) -> tuple[str, int]:
    end = text.find('\"\"\"', i + 3)
    return (text[i:], len(text)) if end == -1 else (text[i : end + 3], end + 3)


def consume_cpp_raw_string(text: str, i: int) -> tuple[str, int] | None:
    for prefix in ("u8R\"", "uR\"", "UR\"", "LR\"", "R\""):
        if not text.startswith(prefix, i):
            continue

        start = i + len(prefix)
        paren = text.find("(", start)

        if paren == -1:
            return None

        delimiter = text[start:paren]

        if len(delimiter) > 16 or any(ch in delimiter for ch in " \\()"):
            return None

        terminator = ")" + delimiter + '"'
        end = text.find(terminator, paren + 1)

        return (
            (text[i:], len(text))
            if end == -1
            else (text[i : end + len(terminator)], end + len(terminator))
        )

    return None


def tokenize_c_family(
    text: str,
    *,
    language: str,
    normalize_literals: bool,
    normalize_identifiers: bool,
) -> list[str]:
    tokens: list[str] = []
    keywords = LANGUAGE_KEYWORDS.get(language, set())
    i = 0

    while i < len(text):
        ch = text[i]

        if ch.isspace():
            i += 1
            continue

        if text.startswith("//", i):
            newline = text.find("\n", i + 2)
            i = len(text) if newline == -1 else newline + 1
            continue

        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = len(text) if end == -1 else end + 2
            continue

        if language == "cpp":
            raw = consume_cpp_raw_string(text, i)
            if raw is not None:
                literal, i = raw
                tokens.append("<STR>" if normalize_literals else literal)
                continue

        if language == "java" and text.startswith('\"\"\"', i):
            literal, i = consume_java_text_block(text, i)
            tokens.append("<STR>" if normalize_literals else literal)
            continue

        if ch in {'"', "'"}:
            literal, i = consume_quoted_literal(text, i, ch)
            tokens.append(("<STR>" if ch == '"' else "<CHAR>") if normalize_literals else literal)
            continue

        number_match = NUMBER_RE.match(text, i)
        if number_match:
            literal = number_match.group(0)
            tokens.append("<NUM>" if normalize_literals else literal)
            i = number_match.end()
            continue

        identifier_match = IDENTIFIER_RE.match(text, i)
        if identifier_match:
            ident = identifier_match.group(0)
            tokens.append("<ID>" if normalize_identifiers and ident not in keywords else ident)
            i = identifier_match.end()
            continue

        for op in OPERATORS:
            if text.startswith(op, i):
                tokens.append(op)
                i += len(op)
                break
        else:
            tokens.append(ch)
            i += 1

    return tokens


def tokenize_generic(
    text: str,
    *,
    normalize_literals: bool,
    normalize_identifiers: bool,
) -> list[str]:
    # Use the C-family lexer to remove comments first, then regex-tokenize the result.
    comment_stripped = " ".join(
        tokenize_c_family(
            text,
            language="generic",
            normalize_literals=False,
            normalize_identifiers=False,
        )
    )
    raw_tokens = [m.group(0) for m in GENERIC_TOKEN_RE.finditer(comment_stripped)]

    if not (normalize_literals or normalize_identifiers):
        return raw_tokens

    normalized: list[str] = []

    for tok in raw_tokens:
        if normalize_literals and (
            tok[:1] in {"'", '"', "`"}
            or re.fullmatch(r"0[xX][0-9A-Fa-f]+|0[bB][01]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", tok)
        ):
            normalized.append("<LIT>")
        elif normalize_identifiers and re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", tok):
            normalized.append("<ID>")
        else:
            normalized.append(tok)

    return normalized


def tokenize_file(
    path: Path,
    *,
    normalize_literals: bool,
    normalize_identifiers: bool,
) -> tuple[list[str], str]:
    text = read_text_lossy(path).replace("\x00", "")
    language = language_from_path(path)

    if language == "python":
        tokens = tokenize_python(
            text,
            normalize_literals=normalize_literals,
            normalize_identifiers=normalize_identifiers,
        )
    elif language in {"c", "cpp", "java"}:
        tokens = tokenize_c_family(
            text,
            language=language,
            normalize_literals=normalize_literals,
            normalize_identifiers=normalize_identifiers,
        )
    else:
        tokens = tokenize_generic(
            text,
            normalize_literals=normalize_literals,
            normalize_identifiers=normalize_identifiers,
        )

    return tokens, language


def write_language_summary(records: list[FileRecord], output_dir: Path) -> Path:
    summary_path = output_dir / "language_summary.csv"
    counts = Counter((r.language, r.status) for r in records)

    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["language", "included", "skipped", "error", "total", "tokens_included"],
        )
        writer.writeheader()

        for language in sorted({r.language for r in records}):
            lang_records = [r for r in records if r.language == language]
            writer.writerow(
                {
                    "language": language,
                    "included": counts[(language, "included")],
                    "skipped": counts[(language, "skipped")],
                    "error": counts[(language, "error")],
                    "total": len(lang_records),
                    "tokens_included": sum(
                        r.token_count for r in lang_records if r.status == "included"
                    ),
                }
            )

    return summary_path


def write_token_file(
    input_dir: Path,
    output_dir: Path,
    files: Iterable[Path],
    *,
    min_tokens: int,
    normalize_literals: bool,
    normalize_identifiers: bool,
) -> tuple[Path, list[FileRecord], int, Path]:
    token_dir = output_dir / "tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    jsonl_gz_path = token_dir / "tokens.jsonl.gz"
    records: list[FileRecord] = []
    included_count = 0

    with gzip.open(jsonl_gz_path, "wt", encoding="utf-8", compresslevel=6) as out:
        for path in files:
            rel = path.resolve().relative_to(input_dir.resolve()).as_posix()
            language = language_from_path(path)

            try:
                tokens, language = tokenize_file(
                    path,
                    normalize_literals=normalize_literals,
                    normalize_identifiers=normalize_identifiers,
                )
                size = path.stat().st_size
                digest = sha256_file(path)

                if len(tokens) < min_tokens:
                    records.append(
                        FileRecord(
                            rel,
                            str(path.resolve()),
                            language,
                            size,
                            digest,
                            len(tokens),
                            "skipped",
                            "too few tokens",
                        )
                    )
                    continue

                out.write(
                    json.dumps(
                        {
                            "filename": rel,
                            "tokens": tokens,
                            "_metadata": {
                                "sha256": digest,
                                "size_bytes": size,
                                "extension": path.suffix.lower(),
                                "language": language,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                records.append(
                    FileRecord(
                        rel,
                        str(path.resolve()),
                        language,
                        size,
                        digest,
                        len(tokens),
                        "included",
                    )
                )
                included_count += 1

            except Exception as exc:
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0

                records.append(
                    FileRecord(
                        rel,
                        str(path.resolve()),
                        language,
                        size,
                        "",
                        0,
                        "error",
                        str(exc),
                    )
                )

    manifest_csv = output_dir / "token_manifest.csv"

    with manifest_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(asdict(FileRecord("", "", "", 0, "", 0, "")).keys()),
        )
        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))

    language_summary_path = write_language_summary(records, output_dir)
    return jsonl_gz_path, records, included_count, language_summary_path


def run_detector(
    csproj: Path,
    token_gz_path: Path,
    output_dir: Path,
    *,
    output_prefix: str,
    key_jaccard_threshold: float,
    jaccard_threshold: float,
    quiet: bool,
    keep_temp: bool,
) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="ndcd-input-"))

    try:
        shutil.copy2(token_gz_path, temp_dir / token_gz_path.name)

        cmd = [
            "dotnet",
            "run",
            "--project",
            str(csproj),
            "--dir",
            str(temp_dir),
            "--key-jaccard-threshold",
            str(key_jaccard_threshold),
            "--jaccard-threshold",
            str(jaccard_threshold),
            output_prefix,
        ]

        result = run_command(cmd, cwd=output_dir, quiet=quiet)

        (output_dir / f"{output_prefix}.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (output_dir / f"{output_prefix}.stderr.txt").write_text(result.stderr, encoding="utf-8")

        if not quiet and result.stdout.strip():
            print(result.stdout.strip())
        if not quiet and result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)

        clusters_path = output_dir / f"{output_prefix}.json"
        log_path = output_dir / f"{output_prefix}.log"

        if not clusters_path.exists():
            clusters_path.write_text("[]\n", encoding="utf-8")

        return clusters_path, log_path if log_path.exists() else None

    finally:
        if keep_temp:
            if not quiet:
                print(f"Kept temporary detector input at: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


def normalize_clusters(raw: object) -> list[list[str]]:
    clusters: list[list[str]] = []

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, list):
                cluster = [str(x) for x in item]
            elif isinstance(item, dict):
                values = (
                    item.get("files")
                    or item.get("Files")
                    or item.get("members")
                    or item.get("Members")
                    or []
                )
                cluster = [str(x) for x in values] if isinstance(values, list) else []
            else:
                cluster = []

            if len(cluster) > 1:
                clusters.append(cluster)

    elif isinstance(raw, dict):
        return normalize_clusters(raw.get("clusters") or raw.get("Clusters") or [])

    return clusters


def analyze_clusters(clusters_path: Path, total_files: int) -> dict:
    clusters = normalize_clusters(json.loads(clusters_path.read_text(encoding="utf-8")))
    duplicate_files = {filename for cluster in clusters for filename in cluster}
    cluster_sizes = [len(c) for c in clusters]
    duplicate_count = len(duplicate_files)
    unique_count = max(total_files - duplicate_count, 0)

    return {
        "total_input_files": total_files,
        "duplicate_cluster_count": len(clusters),
        "duplicate_file_count": duplicate_count,
        "unique_file_count": unique_count,
        "true_uniqueness_score": unique_count / total_files if total_files else 0.0,
        "duplication_rate": duplicate_count / total_files if total_files else 0.0,
        "average_cluster_size": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0.0,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "clusters_path": str(clusters_path),
    }


def write_pair_csv(log_path: Path | None, output_dir: Path) -> Path | None:
    if log_path is None or not log_path.exists():
        return None

    output_csv = output_dir / "duplicate_pairs.csv"

    with log_path.open("r", encoding="utf-8", errors="replace") as src, output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=["file1", "file2", "jaccard_similarity", "key_jaccard_similarity"],
        )
        writer.writeheader()

        for line in src:
            parts = line.strip().rsplit(",", 2)
            if len(parts) != 3:
                continue

            left, jaccard, key_jaccard = parts
            file_parts = left.split(",", 1)

            if len(file_parts) == 2:
                writer.writerow(
                    {
                        "file1": file_parts[0],
                        "file2": file_parts[1],
                        "jaccard_similarity": jaccard,
                        "key_jaccard_similarity": key_jaccard,
                    }
                )

    return output_csv


def _validate_threshold(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _resolve_extensions(config: UniquenessConfig) -> set[str]:
    if config.extensions is not None:
        return {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in config.extensions
        }

    valid_languages = set(LANGUAGE_EXTENSIONS) | {"generic"}
    unknown_languages = set(config.languages) - valid_languages

    if unknown_languages:
        raise ValueError(f"Unknown language(s): {', '.join(sorted(unknown_languages))}")

    return extensions_for_languages(config.languages)


def install_uniqueness_detector(config: UniquenessConfig | None = None) -> Path:
    """
    Install/setup Microsoft's near-duplicate-code-detector without running analysis.

    This clones the detector repository if it is missing, optionally patches it,
    optionally builds it, and returns the detector .csproj path.

    Requires git and dotnet to be available on PATH.
    """
    config = config or UniquenessConfig()
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    return ensure_detector(
        output_dir,
        config.tool_root,
        patch=config.patch_detector,
        build=config.build_detector,
        min_tokens_for_detector=config.detector_min_tokens,
        target_framework=config.target_framework,
        quiet=config.quiet,
    )


def run_uniqueness_detection(
    input_dir: str | Path,
    config: UniquenessConfig | None = None,
) -> UniquenessResult:
    """
    Tokenize a source folder, optionally run Microsoft's near-duplicate-code-detector,
    and return output paths plus metrics.

    Parameters
    ----------
    input_dir:
        Folder containing source files to analyze.

    config:
        Optional UniquenessConfig. If omitted, defaults are used.

    Returns
    -------
    UniquenessResult
        Contains output paths, tokenization records, and detector metrics.

    Raises
    ------
    FileNotFoundError
        If input_dir does not exist or is not a directory.

    RuntimeError
        If no files were tokenized, or if git/dotnet setup fails.

    subprocess.CalledProcessError
        If dotnet build/run fails.
    """
    config = config or UniquenessConfig()

    _validate_threshold("key_jaccard_threshold", config.key_jaccard_threshold)
    _validate_threshold("jaccard_threshold", config.jaccard_threshold)

    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist or is not a directory: {input_dir}")

    if config.max_file_size_mb <= 0:
        raise ValueError("max_file_size_mb must be greater than 0")

    if config.min_tokens < 0:
        raise ValueError("min_tokens must be >= 0")

    if config.detector_min_tokens < 0:
        raise ValueError("detector_min_tokens must be >= 0")

    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = _resolve_extensions(config)
    exclude_dirs = set(DEFAULT_EXCLUDE_DIRS) | set(config.exclude_dirs)
    max_file_size_bytes = int(config.max_file_size_mb * 1024 * 1024)

    files = list(
        iter_source_files(
            input_dir,
            output_dir,
            extensions,
            include_hidden=config.include_hidden,
            follow_symlinks=config.follow_symlinks,
            exclude_dirs=exclude_dirs,
            max_file_size_bytes=max_file_size_bytes,
        )
    )

    token_gz_path, records, included_count, language_summary_path = write_token_file(
        input_dir,
        output_dir,
        files,
        min_tokens=config.min_tokens,
        normalize_literals=config.normalize_literals,
        normalize_identifiers=config.normalize_identifiers,
    )

    skipped_or_errored_count = sum(1 for r in records if r.status != "included")
    manifest_csv = output_dir / "token_manifest.csv"

    result = UniquenessResult(
        output_dir=output_dir,
        token_file=token_gz_path,
        manifest_csv=manifest_csv,
        language_summary_csv=language_summary_path,
        records=records,
        included_count=included_count,
        skipped_or_errored_count=skipped_or_errored_count,
    )

    if included_count == 0:
        raise RuntimeError(
            "No files were tokenized. Adjust languages, extensions, min_tokens, or max_file_size_mb."
        )

    if config.tokens_only:
        return result

    csproj = ensure_detector(
        output_dir,
        config.tool_root,
        patch=config.patch_detector,
        build=config.build_detector,
        min_tokens_for_detector=config.detector_min_tokens,
        target_framework=config.target_framework,
        quiet=config.quiet,
    )

    clusters_path, log_path = run_detector(
        csproj,
        token_gz_path,
        output_dir,
        output_prefix=config.output_prefix,
        key_jaccard_threshold=config.key_jaccard_threshold,
        jaccard_threshold=config.jaccard_threshold,
        quiet=config.quiet,
        keep_temp=config.keep_temp,
    )

    language_counts = Counter(r.language for r in records if r.status == "included")

    metrics = analyze_clusters(clusters_path, included_count)
    metrics.update(
        {
            "candidate_files_discovered": len(files),
            "included_files_tokenized": included_count,
            "skipped_or_errored_files": skipped_or_errored_count,
            "included_by_language": dict(sorted(language_counts.items())),
            "token_file": str(token_gz_path),
            "manifest_csv": str(manifest_csv),
            "language_summary_csv": str(language_summary_path),
            "detector_log": str(log_path) if log_path else None,
            "key_jaccard_threshold": config.key_jaccard_threshold,
            "jaccard_threshold": config.jaccard_threshold,
            "normalize_literals": config.normalize_literals,
            "normalize_identifiers": config.normalize_identifiers,
        }
    )

    metrics_path = output_dir / "uniqueness_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pairs_csv = write_pair_csv(log_path, output_dir)

    result.clusters_path = clusters_path
    result.detector_log = log_path
    result.duplicate_pairs_csv = pairs_csv
    result.metrics_path = metrics_path
    result.metrics = metrics

    return result


__all__ = [
    "FileRecord",
    "UniquenessConfig",
    "UniquenessResult",
    "install_uniqueness_detector",
    "run_uniqueness_detection",
    "ensure_detector",
    "write_token_file",
    "run_detector",
    "analyze_clusters",
]
