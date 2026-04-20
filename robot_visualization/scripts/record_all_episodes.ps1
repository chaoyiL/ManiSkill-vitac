param(
    [string]$ZarrPath = "C:\Users\ruich\Downloads\_0115_bi_pick_and_place_2ver.zarr.zip",
    [string]$OutputDir = "C:\Codes\robot_visualization",
    [int]$EpisodeCount = 169,
    [int]$Fps = 30,
    [string]$Python = "C:\Users\ruich\AppData\Local\Programs\Python\Python311\python.exe"
)

$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Codes\robot_visualization"
$VizScript = Join-Path $RepoRoot "src\viz_3d_enhanced.py"

if (-not (Test-Path $ZarrPath)) {
    Write-Error "Zarr file not found: $ZarrPath"
    exit 1
}

if (-not (Test-Path $VizScript)) {
    Write-Error "Visualizer script not found: $VizScript"
    exit 1
}

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

# record_episode is 0-based; output files are demo001..demo169
for ($i = 0; $i -lt $EpisodeCount; $i++) {
    $fileName = "demo{0:000}.mp4" -f ($i + 1)
    $outPath = Join-Path $OutputDir $fileName
    Write-Host "Recording episode $i -> $outPath"

    & $Python $VizScript $ZarrPath -r --record_episode $i --output_video $outPath --fps $Fps 
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Recording failed on episode $i (exit code $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
}

Write-Host "All recordings complete."