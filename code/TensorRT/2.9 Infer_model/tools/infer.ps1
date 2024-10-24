# how to use:
#   .\tools\infer.ps1 "build\engines\onnx-fp16.engine"

param (
    [String]$inputEngine,
    [String]$tag = ""
)

$PREFIX = [System.IO.Path]::GetFileNameWithoutExtension($inputEngine)

if ($tag -ne "") {
    $PREFIX = "$PREFIX-$tag"
}

$MODE = "infer"
$ONNX_PATH = "models"
$BUILD_PATH = "build"
$ENGINE_PATH = Join-Path $BUILD_PATH "engines"
$LOG_PATH = Join-Path $BUILD_PATH "log\$PREFIX\$MODE"

New-Item -Path $ENGINE_PATH -ItemType Directory -Force
New-Item -Path $LOG_PATH -ItemType Directory -Force

$engineFile = Join-Path $ENGINE_PATH "$PREFIX.engine"
$logFile = Join-Path $LOG_PATH "infer.log"
$outputLog = Join-Path $LOG_PATH "infer_output.log"
$profileLog = Join-Path $LOG_PATH "infer_profile.log"
$layerInfoLog = Join-Path $LOG_PATH "infer_layer_info.log"

$trtexecPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.1.0.27\bin\trtexec.exe"

& $trtexecPath --loadEngine=$engineFile `
              --dumpOutput `
              --dumpProfile `
              --dumpLayerInfo `
              --exportOutput=$outputLog `
              --exportProfile=$profileLog `
              --exportLayerInfo=$layerInfoLog `
              --warmUp=200 `
              --iterations=50 | Out-File $logFile
