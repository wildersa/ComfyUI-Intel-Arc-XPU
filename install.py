"""ComfyUI Manager / CLI installer for ComfyUI-Intel-Arc-XPU (v0.2.0)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TAG_NAME = "v0.2.0"
REPO_SLUG = "wildersa/ComfyUI-Intel-Arc-XPU"
RELEASE_BASE_URL = f"https://github.com/{REPO_SLUG}/releases/download/{TAG_NAME}"

TARGET_TORCH_PREFIX = "2.13.0+xpu"
TARGET_PY_VERSION = (3, 13)

OFFICIAL_PACKAGES: Tuple[Tuple[str, str], ...] = (
    ("comfy-aimdo", "0.6.0"),
    ("comfy-kitchen", "0.2.41"),
)

WHEEL_SPECS: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "omni-xpu-kernel",
        "0.2.0b3+torch213.bmg",
        "omni_xpu_kernel-0.2.0b3+torch213.bmg-cp313-cp313-win_amd64.whl",
        "a5f3d152bce545c517a2e8f1db958c3c14d67238e7467cfa89bbbb859cb9fd25",
    ),
    (
        "comfy-kitchen-xpu-runtime",
        "0.2.41",
        "comfy_kitchen_xpu_runtime-0.2.41-py3-none-any.whl",
        "638756e43526542b625d5a81717bc0721f503e53240416479b42922cde52c605",
    ),
    (
        "comfy-aimdo-xpu-runtime",
        "0.6.0",
        "comfy_aimdo_xpu_runtime-0.6.0-cp39-abi3-win_amd64.whl",
        "6f962d5bdc1f6be61e9ee1d4654b846ff3edea31e4c4fc4640ded6a405756981",
    ),
)


def parse_sha256sums(sums_path: Path) -> Dict[str, str]:
    """Parse SHA256SUMS.txt into a {filename: lowercase_sha256} mapping."""
    hashes: Dict[str, str] = {}
    if not sums_path.is_file():
        return hashes
    for raw_line in sums_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        digest = parts[0].lower()
        filename = parts[-1].lstrip("*")
        if len(digest) == 64:
            hashes[filename] = digest
    return hashes


def resolve_expected_hashes(repo_dir: Path) -> Dict[str, str]:
    """Load expected wheel SHA-256 hashes from SHA256SUMS.txt with built-in fallback."""
    expected = {filename: sha for _, _, filename, sha in WHEEL_SPECS}
    file_hashes = parse_sha256sums(repo_dir / "SHA256SUMS.txt")
    for filename, sha in file_hashes.items():
        if filename in expected and sha != expected[filename]:
            raise RuntimeError(
                f"SHA256SUMS.txt mismatch for {filename}: "
                f"expected {expected[filename]}, found {sha}"
            )
        expected[filename] = sha
    return expected


def compute_sha256(file_path: Path) -> str:
    """Compute lowercase SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def verify_sha256(file_path: Path, expected_sha256: str) -> None:
    """Verify file SHA-256 digest and raise RuntimeError on mismatch."""
    actual = compute_sha256(file_path)
    if actual != expected_sha256.lower():
        raise RuntimeError(
            f"SHA-256 verification failed for {file_path.name}: "
            f"expected {expected_sha256.lower()}, got {actual}"
        )


def get_installed_version(dist_name: str) -> Optional[str]:
    """Return installed distribution version via importlib.metadata, or None."""
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_torch_version() -> Optional[str]:
    """Return installed PyTorch version without importing torch if possible."""
    ver = get_installed_version("torch")
    if ver:
        return ver
    try:
        import torch  # noqa: WPS433

        return str(getattr(torch, "__version__", ""))
    except Exception:
        return None


def check_environment(
    *,
    plat: Optional[str] = None,
    py_ver: Optional[Tuple[int, int]] = None,
    ptr_bytes: Optional[int] = None,
    torch_ver: Optional[str] = None,
) -> Tuple[bool, str]:
    """Verify Windows x64, Python 3.13, and torch==2.13.0+xpu."""
    plat = sys.platform if plat is None else plat
    py_ver = sys.version_info[:2] if py_ver is None else py_ver
    ptr_bytes = struct.calcsize("P") if ptr_bytes is None else ptr_bytes

    if plat != "win32" or ptr_bytes != 8:
        return (
            False,
            f"Skipped: Windows x64 is required (detected platform={plat!r}, "
            f"arch={platform.machine()!r}, {ptr_bytes * 8}-bit).",
        )

    if py_ver != TARGET_PY_VERSION:
        return (
            False,
            f"Skipped: Python 3.13 (cp313) is required for prebuilt wheels; "
            f"detected Python {py_ver[0]}.{py_ver[1]}.",
        )

    if torch_ver is None:
        torch_ver = get_torch_version()
    if not torch_ver or not torch_ver.startswith(TARGET_TORCH_PREFIX):
        return (
            False,
            f"Skipped: PyTorch {TARGET_TORCH_PREFIX} is required "
            f"(detected torch={torch_ver!r}).\n"
            f"Install it first with:\n"
            f'  "{sys.executable}" -m pip install --upgrade '
            f"torch==2.13.0+xpu torchvision==0.28.0+xpu "
            f"--index-url https://download.pytorch.org/whl/xpu",
        )

    return True, f"Environment OK: Windows x64, Python 3.13, torch=={torch_ver}"


def plan_installations(
    installed_versions: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """Return (official_packages_to_install, wheels_to_install)."""
    def _current(name: str) -> Optional[str]:
        if installed_versions is not None:
            return installed_versions.get(name)
        return get_installed_version(name)

    official_needed: List[Tuple[str, str]] = []
    for dist_name, target_ver in OFFICIAL_PACKAGES:
        if _current(dist_name) != target_ver:
            official_needed.append((dist_name, target_ver))

    wheels_needed: List[Tuple[str, str, str]] = []
    for dist_name, target_ver, filename, _ in WHEEL_SPECS:
        if _current(dist_name) != target_ver:
            wheels_needed.append((dist_name, target_ver, filename))

    return official_needed, wheels_needed


def find_uv_executable(python_exe: Path) -> Optional[str]:
    """Locate uv executable if available alongside Python or on PATH."""
    candidates = [
        python_exe.with_name("uv.exe"),
        python_exe.parent.parent.parent / "standalone-env" / "uv.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    which_uv = shutil.which("uv")
    return which_uv


def build_install_command(
    targets: List[str],
    *,
    python_exe: Optional[str] = None,
    uv_exe: Optional[str] = None,
    force_reinstall: bool = False,
) -> List[str]:
    """Build uv pip or python -m pip install command with --no-deps."""
    py = python_exe or sys.executable
    if uv_exe:
        cmd = [uv_exe, "pip", "install", "--python", py, "--no-deps"]
        if force_reinstall:
            cmd.append("--reinstall")
        cmd.extend(targets)
        return cmd
    cmd = [py, "-m", "pip", "install", "--no-deps"]
    if force_reinstall:
        cmd.append("--force-reinstall")
    cmd.extend(targets)
    return cmd


def download_file(url: str, dest_path: Path) -> None:
    """Download a release asset to dest_path atomically."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ComfyUI-Intel-Arc-XPU-Installer/0.2.0"},
    )
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, tmp_path.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1024 * 1024)
        tmp_path.replace(dest_path)
    except urllib.error.URLError as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def obtain_wheel(
    filename: str,
    expected_sha256: str,
    local_wheels_dir: Path,
    temp_dir: Path,
) -> Path:
    """Resolve wheel from local wheels/ directory or download from GitHub Release."""
    local_path = local_wheels_dir / filename
    if local_path.is_file():
        print(f"[ComfyUI-Intel-Arc-XPU] Using local wheel: {local_path}")
        verify_sha256(local_path, expected_sha256)
        return local_path

    url = f"{RELEASE_BASE_URL}/{filename}"
    dest_path = temp_dir / filename
    print(f"[ComfyUI-Intel-Arc-XPU] Downloading {url} ...")
    download_file(url, dest_path)
    verify_sha256(dest_path, expected_sha256)
    print(f"[ComfyUI-Intel-Arc-XPU] Verified SHA-256 for {filename}")
    return dest_path


def print_post_install_reminders(custom_nodes_dir: Optional[Path] = None) -> None:
    """Print Windows configuration reminders."""
    print("")
    print("=== ComfyUI-Intel-Arc-XPU (v0.2.0) Setup Notes ===")
    print("1. Attention backend on Windows:")
    print("   Windows defaults to PyTorch SDPA unless OMNI_ATTN_BACKEND=cute is set.")
    print("   Recommended for Intel Arc BMG: set OMNI_ATTN_BACKEND=cute")
    print("2. DynamicVRAM (AIMDO XPU):")
    print("   Launch ComfyUI with: --enable-dynamic-vram --reserve-vram 4")
    if custom_nodes_dir is not None:
        conflict = custom_nodes_dir / "comfyui-xpu-backend"
        if conflict.exists():
            print(
                f"3. WARNING: Conflicting custom node detected at {conflict}. "
                "Remove or disable comfyui-xpu-backend so it does not conflict with OmniXPU."
            )
            return
    print("3. Conflict check:")
    print("   Avoid keeping custom_nodes/comfyui-xpu-backend alongside this extension.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ComfyUI-Intel-Arc-XPU v0.2.0 installer")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate environment and print planned actions without installing.",
    )
    args = parser.parse_args(argv)

    repo_dir = Path(__file__).resolve().parent
    ok, msg = check_environment()
    print(f"[ComfyUI-Intel-Arc-XPU] {msg}")
    if not ok:
        print(
            "[ComfyUI-Intel-Arc-XPU] Exiting cleanly without modifying your environment."
        )
        return 0

    expected_hashes = resolve_expected_hashes(repo_dir)
    official_needed, wheels_needed = plan_installations()

    if not official_needed and not wheels_needed:
        print(
            "[ComfyUI-Intel-Arc-XPU] All target packages (comfy-aimdo==0.6.0, "
            "comfy-kitchen==0.2.41, omni-xpu-kernel==0.2.0b3+torch213.bmg, "
            "comfy-kitchen-xpu-runtime==0.2.41, comfy-aimdo-xpu-runtime==0.6.0) "
            "are already installed; skipping reinstall."
        )
        print_post_install_reminders(repo_dir.parent)
        return 0

    uv_exe = find_uv_executable(Path(sys.executable))
    local_wheels_dir = repo_dir / "wheels"

    print("[ComfyUI-Intel-Arc-XPU] Planned installs:")
    for dist_name, target_ver in official_needed:
        print(f"  - {dist_name}=={target_ver}")
    for dist_name, target_ver, filename in wheels_needed:
        source = "local wheels/" if (local_wheels_dir / filename).is_file() else RELEASE_BASE_URL
        print(f"  - {dist_name}=={target_ver} ({filename} from {source})")

    if args.dry_run:
        print("[ComfyUI-Intel-Arc-XPU] Dry run complete; no changes made.")
        print_post_install_reminders(repo_dir.parent)
        return 0

    if official_needed:
        pkg_specs = [f"{name}=={ver}" for name, ver in official_needed]
        need_force = any(get_installed_version(name) is not None for name, _ in official_needed)
        cmd = build_install_command(
            pkg_specs,
            python_exe=sys.executable,
            uv_exe=uv_exe,
            force_reinstall=need_force,
        )
        print(f"[ComfyUI-Intel-Arc-XPU] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    if wheels_needed:
        with tempfile.TemporaryDirectory(prefix="omnixpu_wheels_") as tmp:
            temp_dir = Path(tmp)
            wheel_paths: List[str] = []
            need_force = any(
                get_installed_version(dist_name) is not None
                for dist_name, _, _ in wheels_needed
            )
            for _, _, filename in wheels_needed:
                whl = obtain_wheel(
                    filename,
                    expected_hashes[filename],
                    local_wheels_dir,
                    temp_dir,
                )
                wheel_paths.append(str(whl))
            cmd = build_install_command(
                wheel_paths,
                python_exe=sys.executable,
                uv_exe=uv_exe,
                force_reinstall=need_force,
            )
            print(f"[ComfyUI-Intel-Arc-XPU] Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

    print_post_install_reminders(repo_dir.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
