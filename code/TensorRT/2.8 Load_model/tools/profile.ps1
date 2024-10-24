# how to use:
#   .\tools\profile.ps1 "build\engines\onnx-fp16.engine"

param (
    [String]$inputEngine,
    [String]$tag
)

$PREFIX = [System.IO.Path]::GetFileNameWithoutExtension($inputEngine)

if ($tag -ne "") {
    $PREFIX = "$PREFIX-$tag"
}

$MODE = "profile"
$ONNX_PATH = "models"
$BUILD_PATH = "build"
$ENGINE_PATH = Join-Path $BUILD_PATH "engines"
$LOG_PATH = Join-Path $BUILD_PATH "log\$PREFIX\$MODE"

New-Item -Path $ENGINE_PATH -ItemType Directory -Force
New-Item -Path $LOG_PATH -ItemType Directory -Force

$engineFile = Join-Path $ENGINE_PATH "$PREFIX.engine"
$logFile = Join-Path $LOG_PATH "profile.log"
$nsysOutputFile = Join-Path $LOG_PATH $PREFIX

$trtexecPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.1.0.27\bin\trtexec.exe"
$nsysPath = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2023.1.2\target-windows-x64\nsys.exe"

& $nsysPath profile `
    --output=$nsysOutputFile `
    --force-overwrite=true `
    $trtexecPath --loadEngine=$engineFile `
                  --warmUp=0 `
                  --duration=0 `
                  --iterations=20 `
                  --noDataTransfers |
    Out-File $logFile

