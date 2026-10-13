param(
    [string]$OutputRoot = (Join-Path (Split-Path $PSScriptRoot -Parent) 'vlm_image_pairs'),
    [ValidateSet('time_clock_v1', 'public_info_v1')]
    [string[]]$Assets = @('time_clock_v1', 'public_info_v1')
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$repoRoot = Split-Path $PSScriptRoot -Parent
$assetRoot = Join-Path $repoRoot 'ManiSkill/mani_skill/assets/carrot'

function Read-GlbTexture {
    param([string]$Path)

    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ([BitConverter]::ToUInt32($bytes, 0) -ne 0x46546C67) {
        throw "Not a GLB file: $Path"
    }

    $jsonLength = [BitConverter]::ToInt32($bytes, 12)
    $document = [Text.Encoding]::UTF8.GetString($bytes, 20, $jsonLength) | ConvertFrom-Json
    $binStart = 20 + $jsonLength + 8
    $image = $document.images[0]
    $view = $document.bufferViews[$image.bufferView]
    $imageBytes = New-Object byte[] ([int]$view.byteLength)
    [Array]::Copy($bytes, $binStart + [int]$view.byteOffset, $imageBytes, 0, $imageBytes.Length)

    $stream = [System.IO.MemoryStream]::new($imageBytes, $false)
    try {
        $decoded = [System.Drawing.Image]::FromStream($stream)
        try {
            return [System.Drawing.Bitmap]::new($decoded)
        } finally {
            $decoded.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

function Write-PairImage {
    param(
        [string]$AssetDir,
        [string]$LeftName,
        [string]$RightName,
        [string]$Path,
        [bool]$RotateTiles
    )

    $leftPath = Join-Path $AssetDir "shapes/$LeftName/textured.glb"
    $rightPath = Join-Path $AssetDir "shapes/$RightName/textured.glb"
    $left = Read-GlbTexture $leftPath
    $right = Read-GlbTexture $rightPath
    if ($RotateTiles) {
        $left.RotateFlip([System.Drawing.RotateFlipType]::Rotate90FlipNone)
        $right.RotateFlip([System.Drawing.RotateFlipType]::Rotate90FlipNone)
        $left.RotateFlip([System.Drawing.RotateFlipType]::RotateNoneFlipX)
        $right.RotateFlip([System.Drawing.RotateFlipType]::RotateNoneFlipX)
    }
    $canvas = [System.Drawing.Bitmap]::new(1600, 800)
    $graphics = [System.Drawing.Graphics]::FromImage($canvas)
    try {
        $graphics.Clear([System.Drawing.Color]::White)
        $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $graphics.DrawImage($left, [System.Drawing.Rectangle]::new(40, 40, 720, 720))
        $graphics.DrawImage($right, [System.Drawing.Rectangle]::new(840, 40, 720, 720))
        $canvas.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $graphics.Dispose()
        $canvas.Dispose()
        $left.Dispose()
        $right.Dispose()
    }
}

foreach ($asset in $Assets) {
    $assetDir = Join-Path $assetRoot $asset
    $outputDir = Join-Path $OutputRoot $asset
    [System.IO.Directory]::CreateDirectory($outputDir) | Out-Null
    $pairs = Get-Content -LiteralPath (Join-Path $assetDir 'pairs.json') -Raw | ConvertFrom-Json

    foreach ($pair in $pairs) {
        $prefix = 'q{0:D2}' -f [int]$pair.question_id
        $rotateTiles = ($asset -eq 'public_info_v1')
        Write-PairImage $assetDir $pair.left $pair.right (Join-Path $outputDir "${prefix}_noswap.png") $rotateTiles
        Write-PairImage $assetDir $pair.right $pair.left (Join-Path $outputDir "${prefix}_swap.png") $rotateTiles
    }
    Write-Output "$asset : $($pairs.Count * 2) images -> $outputDir"
}
