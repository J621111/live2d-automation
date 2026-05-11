from __future__ import annotations

import json
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]


def escape_send_keys(value: str) -> str:
    return value.replace('"', '""')


def escape_powershell_string(value: str) -> str:
    return value.replace('"', '`"')


def activation_fragments(profile: JsonDict) -> list[str]:
    title = str(profile.get("window_title_contains", "Cubism Editor")).strip()
    fragments = [title]
    fragments.extend(
        str(fragment).strip()
        for fragment in profile.get("window_probe_candidates", [])
        if str(fragment).strip()
    )
    unique_fragments: list[str] = []
    for fragment in fragments:
        if fragment and fragment not in unique_fragments:
            unique_fragments.append(fragment)
    return unique_fragments


def executable_menu_sequence(raw_sequence: object) -> list[JsonDict]:
    if not isinstance(raw_sequence, list):
        return []
    return [
        entry
        for entry in raw_sequence
        if isinstance(entry, dict) and str(entry.get("keys", "")).strip()
    ]


def has_apply_template_invocation(profile: JsonDict) -> bool:
    shortcut = str(profile.get("template_shortcut", "")).strip()
    menu_sequence = executable_menu_sequence(profile.get("template_menu_sequence", []))
    return bool(shortcut or menu_sequence)


def has_export_invocation(profile: JsonDict) -> bool:
    shortcut = str(profile.get("export_shortcut", "")).strip()
    menu_sequence = executable_menu_sequence(profile.get("export_menu_sequence", []))
    return bool(shortcut or menu_sequence)


def dialog_sequence_script(action: str, profile: JsonDict) -> str:
    sequences = dict(profile.get("known_dialog_sequences") or {}).get(action, [])
    lines: list[str] = []
    for entry in sequences:
        if not isinstance(entry, dict):
            continue
        keys = str(entry.get("keys", "")).strip()
        if not keys:
            continue
        wait_ms = int(float(entry.get("wait_seconds", 0.35)) * 1000)
        escaped_keys = escape_send_keys(keys)
        lines.append(f"Start-Sleep -Milliseconds {wait_ms}\n")
        lines.append(f'$wshell.SendKeys("{escaped_keys}")\n')
    return "".join(lines)


def window_activation_helper_script(profile: JsonDict) -> str:
    preferred = escape_powershell_string(
        str(profile.get("window_title_contains", "Cubism Editor")).strip()
    )
    activation_timeout_ms = int(float(profile.get("window_activation_timeout_seconds", 8.0)) * 1000)
    poll_interval_ms = int(float(profile.get("window_activation_poll_seconds", 0.25)) * 1000)
    return (
        "function Get-ControllerWindows {\n"
        "    Get-Process | Where-Object {\n"
        "        $_.MainWindowTitle\n"
        "    } | ForEach-Object {\n"
        "        [PSCustomObject]@{\n"
        "            ProcessId = $_.Id\n"
        "            ProcessName = $_.ProcessName\n"
        "            Title = $_.MainWindowTitle\n"
        "        }\n"
        "    }\n"
        "}\n"
        "function Select-PreferredControllerWindow {\n"
        "    param(\n"
        "        [object[]]$Matches,\n"
        "        [string]$PreferredTitle\n"
        "    )\n"
        "    if (-not $Matches -or $Matches.Count -eq 0) {\n"
        "        return $null\n"
        "    }\n"
        "    $documentMatches = @($Matches | Where-Object {\n"
        "        $_.Title -match '-\\s*\\S+$'\n"
        "    })\n"
        "    if ($documentMatches.Count -gt 0) {\n"
        "        return ($documentMatches |\n"
        "            Sort-Object TitleLength -Descending |\n"
        "            Select-Object -First 1)\n"
        "    }\n"
        "    $exact = @($Matches | Where-Object { $_.Title -eq $PreferredTitle })\n"
        "    if ($exact.Count -gt 0) {\n"
        "        return ($exact |\n"
        "            Sort-Object TitleLength -Descending |\n"
        "            Select-Object -First 1)\n"
        "    }\n"
        "    $editorMatches = @($Matches | Where-Object {\n"
        '        $_.Title -like "*Cubism*" -or $_.Title -like "*Editor*"\n'
        "    })\n"
        "    if ($editorMatches.Count -gt 0) {\n"
        "        return ($editorMatches |\n"
        "            Sort-Object TitleLength -Descending |\n"
        "            Select-Object -First 1)\n"
        "    }\n"
        "    return ($Matches |\n"
        "        Sort-Object TitleLength -Descending |\n"
        "        Select-Object -First 1)\n"
        "}\n"
        "function Activate-ControllerWindow {\n"
        "    param([string[]]$Fragments)\n"
        f"    $deadline = (Get-Date).AddMilliseconds({activation_timeout_ms})\n"
        f"    $pollInterval = {poll_interval_ms}\n"
        "    $windows = @()\n"
        "    $matches = @()\n"
        "    $target = $null\n"
        "    $activated = $false\n"
        f'    $preferredTitle = "{preferred}"\n'
        "    $wshell = New-Object -ComObject WScript.Shell\n"
        "    do {\n"
        "        $windows = @(Get-ControllerWindows)\n"
        "        $matches = @($windows | Where-Object {\n"
        "            $windowTitle = $_.Title\n"
        '            $Fragments | Where-Object { $_ -and $windowTitle -like "*$_*" }\n'
        "        })\n"
        "        $matches = @($matches | ForEach-Object {\n"
        "            $_ |\n"
        "                Add-Member -NotePropertyName TitleLength `\n"
        "                    -NotePropertyValue $_.Title.Length -PassThru\n"
        "        })\n"
        "        $target = Select-PreferredControllerWindow `\n"
        "            -Matches $matches `\n"
        "            -PreferredTitle $preferredTitle\n"
        "        if ($null -ne $target) {\n"
        "            $activated = $wshell.AppActivate([int]$target.ProcessId)\n"
        "            if (-not $activated -and $target.Title) {\n"
        "                $activated = $wshell.AppActivate([string]$target.Title)\n"
        "            }\n"
        "            if (-not $activated -and $preferredTitle) {\n"
        "                $activated = $wshell.AppActivate([string]$preferredTitle)\n"
        "            }\n"
        "            if ($activated) { break }\n"
        "        }\n"
        "        Start-Sleep -Milliseconds $pollInterval\n"
        "    } while ((Get-Date) -lt $deadline)\n"
        "    return [PSCustomObject]@{\n"
        "        Windows = $windows\n"
        "        Matches = $matches\n"
        "        Target = $target\n"
        "        Activated = $activated\n"
        "    }\n"
        "}\n"
    )


def launch_script(editor_path: Path, profile: JsonDict) -> str:
    fragments = activation_fragments(profile)
    fragment_json = json.dumps(fragments, ensure_ascii=False).replace('"', '`"')
    delay_ms = int(float(profile.get("launch_wait_seconds", 2.0)) * 1000)
    return (
        '$ErrorActionPreference = "Stop"\n'
        + window_activation_helper_script(profile)
        + f'$activationFragments = ConvertFrom-Json "{fragment_json}"\n'
        f'Start-Process -FilePath "{editor_path}" | Out-Null\n'
        f"Start-Sleep -Milliseconds {delay_ms}\n"
        "$null = Activate-ControllerWindow -Fragments $activationFragments\n"
    )


def import_via_launch_argument_script(
    editor_path: Path,
    psd_path: Path,
    profile: JsonDict,
) -> str:
    delay_ms = int(float(profile.get("dialog_wait_seconds", 0.8)) * 1000)
    return (
        '$ErrorActionPreference = "Stop"\n'
        f'Start-Process -FilePath "{editor_path}" -ArgumentList \'"{psd_path}"\' | Out-Null\n'
        f"Start-Sleep -Milliseconds {delay_ms}\n"
    )


def apply_template_script(template_id: str, profile: JsonDict) -> str:
    fragments = activation_fragments(profile)
    fragment_json = json.dumps(fragments, ensure_ascii=False).replace('"', '`"')
    shortcut = str(profile.get("template_shortcut", "")).strip()
    menu_sequence = executable_menu_sequence(profile.get("template_menu_sequence", []))
    activation_ms = int(float(profile.get("activation_wait_seconds", 1.0)) * 1000)
    dialog_ms = int(float(profile.get("template_dialog_wait_seconds", 0.8)) * 1000)
    escaped_template = template_id.replace("{", "{{").replace("}", "}}")
    lines = ['$ErrorActionPreference = "Stop"\n']
    lines.append(window_activation_helper_script(profile))
    lines.append(f'$activationFragments = ConvertFrom-Json "{fragment_json}"\n')
    lines.append("$windowState = Activate-ControllerWindow -Fragments $activationFragments\n")
    lines.append("$wshell = New-Object -ComObject WScript.Shell\n")
    lines.append('if (-not $windowState.Activated) { throw "Cubism window not found." }\n')
    lines.append(f"Start-Sleep -Milliseconds {activation_ms}\n")
    if menu_sequence:
        for entry in menu_sequence:
            keys = str(entry.get("keys", "")).strip()
            if not keys:
                continue
            wait_ms = int(float(entry.get("wait_seconds", 0.25)) * 1000)
            escaped_keys = escape_send_keys(keys)
            lines.append(f'$wshell.SendKeys("{escaped_keys}")\n')
            lines.append(f"Start-Sleep -Milliseconds {wait_ms}\n")
    elif shortcut:
        lines.append(f'$wshell.SendKeys("{shortcut}")\n')
        lines.append(f"Start-Sleep -Milliseconds {dialog_ms}\n")
    lines.append(f'$wshell.SendKeys("{escaped_template}")\n')
    lines.append("Start-Sleep -Milliseconds 200\n")
    lines.append('$wshell.SendKeys("~")\n')
    return "".join(lines) + dialog_sequence_script("apply_template", profile)


def export_script(output_dir: Path, model_name: str, profile: JsonDict) -> str:
    fragments = activation_fragments(profile)
    fragment_json = json.dumps(fragments, ensure_ascii=False).replace('"', '`"')
    shortcut = str(profile.get("export_shortcut", "^+e")).strip()
    menu_sequence = executable_menu_sequence(profile.get("export_menu_sequence", []))
    activation_ms = int(float(profile.get("activation_wait_seconds", 1.0)) * 1000)
    dialog_ms = int(float(profile.get("export_dialog_wait_seconds", 1.0)) * 1000)
    escaped_output = str(output_dir).replace("{", "{{").replace("}", "}}")
    escaped_name = model_name.replace("{", "{{").replace("}", "}}")
    lines = ['$ErrorActionPreference = "Stop"\n']
    lines.append(window_activation_helper_script(profile))
    lines.append(f'$activationFragments = ConvertFrom-Json "{fragment_json}"\n')
    lines.append("$windowState = Activate-ControllerWindow -Fragments $activationFragments\n")
    lines.append("$wshell = New-Object -ComObject WScript.Shell\n")
    lines.append('if (-not $windowState.Activated) { throw "Cubism window not found." }\n')
    lines.append(f"Start-Sleep -Milliseconds {activation_ms}\n")
    if menu_sequence:
        for entry in menu_sequence:
            keys = str(entry.get("keys", "")).strip()
            if not keys:
                continue
            wait_ms = int(float(entry.get("wait_seconds", 0.25)) * 1000)
            escaped_keys = escape_send_keys(keys)
            lines.append(f'$wshell.SendKeys("{escaped_keys}")\n')
            lines.append(f"Start-Sleep -Milliseconds {wait_ms}\n")
    else:
        lines.append(f'$wshell.SendKeys("{shortcut}")\n')
        lines.append(f"Start-Sleep -Milliseconds {dialog_ms}\n")
    extra_dialogs = dialog_sequence_script("export_embedded_data", profile)
    if extra_dialogs:
        lines.append(extra_dialogs)
    lines.append(f'$wshell.SendKeys("{escaped_output}")\n')
    lines.append("Start-Sleep -Milliseconds 200\n")
    lines.append('$wshell.SendKeys("~")\n')
    lines.append(f"Start-Sleep -Milliseconds {dialog_ms}\n")
    lines.append(f'$wshell.SendKeys("{escaped_name}")\n')
    lines.append("Start-Sleep -Milliseconds 200\n")
    lines.append('$wshell.SendKeys("~")\n')
    return "".join(lines)


def import_script(psd_path: Path, profile: JsonDict) -> str:
    fragments = activation_fragments(profile)
    fragment_json = json.dumps(fragments, ensure_ascii=False).replace('"', '`"')
    shortcut = str(profile.get("import_shortcut", "^o"))
    activation_ms = int(float(profile.get("activation_wait_seconds", 1.0)) * 1000)
    dialog_ms = int(float(profile.get("dialog_wait_seconds", 0.8)) * 1000)
    escaped_psd = str(psd_path).replace("{", "{{").replace("}", "}}")
    script = (
        '$ErrorActionPreference = "Stop"\n'
        + window_activation_helper_script(profile)
        + f'$activationFragments = ConvertFrom-Json "{fragment_json}"\n'
        "$windowState = Activate-ControllerWindow -Fragments $activationFragments\n"
        "$wshell = New-Object -ComObject WScript.Shell\n"
        'if (-not $windowState.Activated) { throw "Cubism window not found." }\n'
        f"Start-Sleep -Milliseconds {activation_ms}\n"
        f'$wshell.SendKeys("{shortcut}")\n'
        f"Start-Sleep -Milliseconds {dialog_ms}\n"
        f'$wshell.SendKeys("{escaped_psd}")\n'
        "Start-Sleep -Milliseconds 200\n"
        '$wshell.SendKeys("~")\n'
    )
    return script + dialog_sequence_script("import_psd", profile)


def window_probe_script(profile: JsonDict, extra_fragments: list[str] | None = None) -> str:
    title = str(profile.get("window_title_contains", "Cubism Editor"))
    fragments = activation_fragments(profile)
    for fragment in extra_fragments or []:
        normalized = str(fragment).strip()
        if normalized and normalized not in fragments:
            fragments.append(normalized)
    fragment_json = json.dumps(fragments, ensure_ascii=False).replace('"', '`"')
    return (
        '$ErrorActionPreference = "Stop"\n'
        + window_activation_helper_script(profile)
        + f'$fragments = ConvertFrom-Json "{fragment_json}"\n'
        "$windowState = Activate-ControllerWindow -Fragments $fragments\n"
        "$result = [PSCustomObject]@{\n"
        f'    target = "{escape_powershell_string(title)}"\n'
        "    matched_titles = @($windowState.Matches | ForEach-Object { $_.Title })\n"
        "    diagnostics = @($windowState.Matches)\n"
        "    all_titles = @($windowState.Windows | ForEach-Object { $_.Title })\n"
        "    all_diagnostics = @($windowState.Windows)\n"
        "}\n"
        "$result | ConvertTo-Json -Depth 4 -Compress | Write-Output\n"
        "if ($windowState.Activated) { exit 0 } else { exit 3 }\n"
    )


def parse_probe_stdout(stdout: str) -> JsonDict:
    if not stdout:
        return {}
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return {}
    candidate = lines[-1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    extracted: JsonDict = {}
    if "matched_titles" in parsed and isinstance(parsed["matched_titles"], list):
        extracted["matched_titles"] = parsed["matched_titles"]
    if "diagnostics" in parsed and isinstance(parsed["diagnostics"], list):
        extracted["diagnostics"] = parsed["diagnostics"]
    if "all_titles" in parsed and isinstance(parsed["all_titles"], list):
        extracted["all_titles"] = parsed["all_titles"]
    if "all_diagnostics" in parsed and isinstance(parsed["all_diagnostics"], list):
        extracted["all_diagnostics"] = parsed["all_diagnostics"]
    if "target" in parsed:
        extracted["target"] = parsed["target"]
    return extracted


def capture_script(
    screenshot_path: Path,
    profile: JsonDict,
    wait_key: str = "failure_capture_wait_seconds",
) -> str:
    delay_ms = int(float(profile.get(wait_key, 0.2)) * 1000)
    escaped_path = str(screenshot_path).replace("'", "''")
    return (
        '$ErrorActionPreference = "Stop"\n'
        "Add-Type -AssemblyName System.Windows.Forms\n"
        "Add-Type -AssemblyName System.Drawing\n"
        f"Start-Sleep -Milliseconds {delay_ms}\n"
        "$bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen\n"
        "$bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height\n"
        "$graphics = [System.Drawing.Graphics]::FromImage($bitmap)\n"
        "$graphics.CopyFromScreen(\n"
        "    $bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size\n"
        ")\n"
        f"$bitmap.Save('{escaped_path}', [System.Drawing.Imaging.ImageFormat]::Png)\n"
        "$graphics.Dispose()\n"
        "$bitmap.Dispose()\n"
    )
