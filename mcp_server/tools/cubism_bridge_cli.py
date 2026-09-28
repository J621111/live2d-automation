"""Compile, self-test, or attach a local Swing bridge using an installed Cubism JVM."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import plistlib
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from .cubism_paths import session_directory, state_directory


@dataclass(frozen=True)
class CubismInstallation:
    """Runtime-only paths discovered from an installed editor distribution."""

    root: Path
    app: Path
    executable: Path
    jvm: Path
    json_jar: Path


def bridge_source() -> Path:
    """Locate Java source shipped with the installed Python package."""
    return Path(__file__).resolve().parents[1] / "java/SwingBridge.java"


def discover_installation(editor_root: str | Path | None = None) -> CubismInstallation:
    """Resolve an explicit app/distribution or one unambiguous macOS installation."""
    configured = editor_root or os.environ.get("LIVE2D_CUBISM_HOME")
    if configured:
        candidates = [Path(configured).expanduser().resolve()]
    else:
        candidates = [
            path
            for base in [Path("/Applications"), Path.home() / "Applications"]
            for path in sorted(base.glob("Live2D Cubism*"))
        ]
    found = []
    for candidate in candidates:
        root = candidate.parent if candidate.suffix == ".app" else candidate
        apps = [candidate] if candidate.suffix == ".app" else sorted(root.glob("*.app"))
        for app in apps:
            try:
                info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
            except (OSError, plistlib.InvalidFileException, ValueError):
                continue
            if info.get("CFBundleExecutable") != "CubismEditor":
                continue
            executable = app / "Contents/MacOS/CubismEditor"
            jvm = root / "jre/Contents/Home/lib/server/libjvm.dylib"
            jars = sorted((root / "res").glob("json-simple*.jar"))
            if executable.is_file() and jvm.is_file() and len(jars) == 1:
                found.append(CubismInstallation(root, app, executable, jvm, jars[0]))
    if len(found) != 1:
        raise RuntimeError(
            "Expected one Cubism installation with its JVM and json-simple; specify --editor-root"
        )
    return found[0]


def verified_editor_pid(
    installation: CubismInstallation,
    pid: int | None = None,
    *,
    run_command=subprocess.run,
) -> int:
    """Require a current same-user process running the exact selected editor executable."""
    result = run_command(
        ["/bin/ps", "-A", "-o", "pid=,uid=,comm="],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("Unable to verify the current Cubism Editor process")
    matches = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        if int(parts[1]) == os.getuid() and parts[2] == str(installation.executable):
            matches.append(int(parts[0]))
    if pid is not None:
        if pid not in matches:
            raise RuntimeError("Requested PID is not a verified same-user Cubism Editor")
        return pid
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one running Cubism Editor; specify --pid")
    return matches[0]


def _private_directory(path: Path) -> Path:
    path = path.expanduser().absolute()
    if path.is_symlink():
        raise RuntimeError("Unsafe bridge directory: symbolic link")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError("Bridge directories must be private to the current user (0700)")
    return path


def _start_java(installation: CubismInstallation, classes: Path):
    try:
        jpype = importlib.import_module("jpype")
    except ImportError as exc:
        raise RuntimeError(
            "Install live2d-automation[macos-cubism] to compile or attach the bridge"
        ) from exc
    if not jpype.isJVMStarted():
        jpype.startJVM(
            str(installation.jvm),
            "-Djava.awt.headless=true",
            "--add-opens=java.desktop/java.awt=ALL-UNNAMED",
            classpath=[str(installation.json_jar), str(classes)],
        )
    else:
        jpype.addClassPath(str(installation.json_jar))
        jpype.addClassPath(str(classes))
    return jpype


def _compile(jpype, installation, classes, sources):
    compiler = jpype.JClass("javax.tools.ToolProvider").getSystemJavaCompiler()
    if compiler is None:
        raise RuntimeError("The selected installation JVM does not provide a Java compiler")
    code = compiler.run(
        None,
        None,
        None,
        "-classpath",
        os.pathsep.join([str(installation.json_jar), str(classes)]),
        "-d",
        str(classes),
        *[str(path) for path in sources],
    )
    if code:
        raise RuntimeError("Swing bridge compilation failed")


def _self_test(jpype) -> None:
    bridge = jpype.JClass("live2d.automation.SwingBridge")
    results = []

    def checks():
        field = jpype.JClass("javax.swing.JTextField")("before")
        view = bridge.snapshot(field, 0)
        command = jpype.JClass("org.json.simple.JSONObject")()
        command.put("action", "set_text")
        command.put("id", view.get("id"))
        command.put("value", "after")
        bridge.perform(command)
        results.append(str(field.getText()) == "after")

    # These widgets are never realized; no desktop event loop is needed for the headless smoke.
    checks()
    if results != [True]:
        raise RuntimeError("Swing self-test did not update the real text widget")


def main(argv: list[str] | None = None) -> int:
    """Run a package-safe bridge operation; attachment requires a verified current editor PID."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["compile", "self-test", "attach"])
    parser.add_argument("--editor-root", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--session-dir", type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument(
        "--test-source", type=Path, help="Optional checkout Java harness for self-test"
    )
    args = parser.parse_args(argv)
    try:
        if os.name != "posix":
            raise RuntimeError("The Cubism attachment CLI requires a macOS/POSIX installation")
        if args.timeout <= 0 or not math.isfinite(args.timeout):
            raise ValueError("Timeout must be finite and positive")
        if args.test_source and args.command != "self-test":
            raise ValueError("--test-source is only supported with self-test")
        installation = discover_installation(args.editor_root)
        pid = verified_editor_pid(installation, args.pid) if args.command == "attach" else None
        build = _private_directory(args.build_dir or state_directory() / "swing-bridge")
        classes = _private_directory(build / "classes")
        jpype = _start_java(installation, classes)
        source = bridge_source()
        if args.command != "attach":
            sources = [source] + ([args.test_source.resolve()] if args.test_source else [])
            _compile(jpype, installation, classes, sources)
            if args.command == "self-test":
                _self_test(jpype)
                if args.test_source:
                    jpype.JClass("live2d.automation.SwingBridgeTest").main([])
            print(
                json.dumps(
                    {"status": "success", "operation": args.command, "classes": str(classes)}
                )
            )
            return 0

        from .cubism_swing import _artifact, _read_json, request

        session = _private_directory(args.session_dir or session_directory())
        directory_fd = os.open(session, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            if _artifact(directory_fd, "request.json") is not None:
                raise RuntimeError("A pending bridge request must be inspected before attaching")
            try:
                ready = _read_json(directory_fd, "ready.json", 4096)
            except FileNotFoundError:
                ready = None
            if ready:
                if ready.get("pid") != pid:
                    raise RuntimeError(
                        "Existing bridge session belongs to another or stale editor process"
                    )
                request({"action": "shutdown"}, args.timeout, session_dir=session)
                deadline = time.monotonic() + args.timeout
                while (session / "ready.json").exists() and time.monotonic() < deadline:
                    time.sleep(0.025)
                if (session / "ready.json").exists():
                    raise RuntimeError("Previous bridge did not stop; session retained")
        finally:
            os.close(directory_fd)
        class_name = "SwingBridge_" + uuid.uuid4().hex
        generated = build / f"{class_name}.java"
        generated.write_text(
            source.read_text().replace("class SwingBridge {", f"class {class_name} {{")
        )
        _compile(jpype, installation, classes, [generated])
        agent = build / f"{class_name}.jar"
        with ZipFile(agent, "w") as archive:
            archive.writestr(
                "META-INF/MANIFEST.MF",
                f"Manifest-Version: 1.0\nAgent-Class: live2d.automation.{class_name}\n\n",
            )
            for compiled in classes.rglob(f"{class_name}*.class"):
                archive.write(compiled, compiled.relative_to(classes).as_posix())
        verified_editor_pid(installation, pid)
        vm = jpype.JClass("com.sun.tools.attach.VirtualMachine").attach(str(pid))
        try:
            vm.loadAgent(str(agent), str(session))
        finally:
            vm.detach()
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            directory_fd = os.open(session, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                try:
                    ready = _read_json(directory_fd, "ready.json", 4096)
                except FileNotFoundError:
                    ready = None
                if ready and ready.get("pid") == pid:
                    print(
                        json.dumps({"status": "attached", "pid": pid, "session_dir": str(session)})
                    )
                    return 0
            finally:
                os.close(directory_fd)
            time.sleep(0.025)
        raise TimeoutError("Attached bridge did not become ready; inspect the private session")
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        print(f"Cubism bridge failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
