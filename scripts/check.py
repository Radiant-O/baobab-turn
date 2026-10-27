"""Everything CI would have run, on your own machine, for free.

    python scripts/check.py           # the full set
    python scripts/check.py --fast    # skip the wheel build

Hosted CI is not part of this project's workflow, so this script is the gate
rather than a convenience: if it passes, the branch is good to push.

It deliberately includes the wheel build. The package was unbuildable for
weeks and nothing noticed, because every install anyone ever did was
`pip install -e .` and editable installs skip wheel packaging entirely. The
first person to find out would have been whoever ran `pip install
baobab-turn`. Building is not enough either -- the failure that matters is a
wheel that builds cleanly and ships no language packs, which only shows up
when something tries to load one. So the wheel is installed into a throwaway
environment and asked to do real work.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(label: str, argv: list[str], cwd: Path | None = None) -> bool:
    print(f"\n{'=' * 64}\n  {label}\n{'=' * 64}")
    result = subprocess.run(argv, cwd=cwd or ROOT)
    ok = result.returncode == 0
    print(f"  -> {'ok' if ok else 'FAILED'}")
    return ok


def check_wheel() -> bool:
    """Build the wheel, install it somewhere clean, and load a pack from it."""
    print(f"\n{'=' * 64}\n  wheel builds, installs, and loads a pack\n{'=' * 64}")

    tmp = Path(tempfile.mkdtemp(prefix="baobab-wheel-"))
    try:
        built = subprocess.run(
            [PY, "-m", "build", "--wheel", "--outdir", str(tmp / "dist")],
            cwd=ROOT,
        )
        if built.returncode != 0:
            print("  -> FAILED (build). If `build` is missing: pip install build")
            return False

        wheels = sorted((tmp / "dist").glob("*.whl"))
        if not wheels:
            print("  -> FAILED (no wheel produced)")
            return False
        print(f"  built {wheels[-1].name}")

        env_dir = tmp / "venv"
        venv.create(env_dir, with_pip=True)
        bin_dir = "Scripts" if sys.platform == "win32" else "bin"
        py = env_dir / bin_dir / ("python.exe" if sys.platform == "win32" else "python")

        installed = subprocess.run(
            [str(py), "-m", "pip", "install", "--quiet", str(wheels[-1])]
        )
        if installed.returncode != 0:
            print("  -> FAILED (install)")
            return False

        # Run from somewhere else entirely, so a successful import cannot be
        # the source tree answering on the wheel's behalf.
        probe = (
            "from baobab_turn import resolve, guard, GuardInput, MarkerIndex;"
            "p = resolve('pcm');"
            "assert p is not None, 'pcm pack missing from the wheel';"
            "assert p.yield_markers, 'pack loaded but carries no markers';"
            "r = guard(GuardInput(p_inner=0.75, transcript_tail='e don finish abi',"
            " is_final=True), index=MarkerIndex(p));"
            "assert r.p_adjusted > 0.75, f'guard did not fire: {r.reasons}';"
            "print(f'  {p.name}: {len(p.yield_markers)} yield markers, p={r.p_adjusted:.3f}')"
        )
        loaded = subprocess.run([str(py), "-c", probe], cwd=tmp)
        ok = loaded.returncode == 0
        print(f"  -> {'ok' if ok else 'FAILED (pack did not load from the wheel)'}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true",
                    help="skip the wheel build, which is the slow part")
    args = ap.parse_args()

    results: list[tuple[str, bool]] = [
        ("lint", run("ruff", [PY, "-m", "ruff", "check", "."])),
        ("tests", run("pytest", [PY, "-m", "pytest", "-q"])),
        # Its own step as well as part of the suite, because this is the
        # architectural boundary and a failure here means something larger
        # than a broken test.
        ("core purity", run(
            "core/ imports no framework",
            [PY, "-m", "pytest", "tests/test_core_purity.py", "-q"],
        )),
    ]
    if not args.fast:
        results.append(("wheel", check_wheel()))

    print(f"\n{'=' * 64}")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}")
        return 1
    print("\nall good" + (" (wheel skipped)" if args.fast else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
