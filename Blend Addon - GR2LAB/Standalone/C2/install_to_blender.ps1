# Sync GR2 LAB addon into Blender 5.2 user addons + pack GR2_LAB_x.y.z.zip
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Src = Join-Path $Root "extracted\gr2lab_blender"
$Dst = "E:\Users\eloct\AppData\Roaming\Blender Foundation\Blender\5.2\scripts\addons\gr2lab_blender"

if (-not (Test-Path $Src)) {
    Write-Error "Source not found: $Src"
}
if (-not (Test-Path (Split-Path $Dst -Parent))) {
    Write-Error "Blender addons folder missing: $(Split-Path $Dst -Parent)"
}

New-Item -ItemType Directory -Force -Path $Dst | Out-Null
# /XD skips caches; /E copies subdirs. Faster than full MIR on huge trees.
robocopy $Src $Dst /E /R:2 /W:1 /XD __pycache__ .git /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
if ($LASTEXITCODE -ge 16) {
    Write-Error "robocopy failed with exit code $LASTEXITCODE"
}
if ($LASTEXITCODE -ge 8) {
    Write-Warning "robocopy exit $LASTEXITCODE (some files were locked; zip still built from source)"
}

$verMatch = (Get-Content (Join-Path $Src "__init__.py") -Raw | Select-String 'version": \((\d+), (\d+), (\d+)\)').Matches
$v = $null
if ($verMatch) {
    $v = "$($verMatch.Groups[1].Value).$($verMatch.Groups[2].Value).$($verMatch.Groups[3].Value)"
    Write-Host "Installed gr2lab_blender $v -> $Dst"
} else {
    Write-Host "Installed gr2lab_blender -> $Dst"
}

$zipName = if ($v) { "GR2_LAB_$v.zip" } else { "GR2_LAB.zip" }
$zipPath = Join-Path $Root $zipName
$py = @"
import zipfile
from pathlib import Path

root = Path(r'$Root')
src = root / 'extracted' / 'gr2lab_blender'
zip_path = Path(r'$zipPath')
if zip_path.exists():
    zip_path.unlink()
count = 0
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for p in src.rglob('*'):
        if not p.is_file():
            continue
        if '__pycache__' in p.parts or p.suffix == '.pyc':
            continue
        arc = 'gr2lab_blender/' + p.relative_to(src).as_posix()
        zf.write(p, arc)
        count += 1
size_kib = zip_path.stat().st_size // 1024
print(f'Packed {zip_path.name} ({count} files, {size_kib} KiB)')
"@
python -c $py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Release zip failed (python exit $LASTEXITCODE)"
}
