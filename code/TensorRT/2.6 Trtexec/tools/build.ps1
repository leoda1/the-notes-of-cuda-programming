# how to use:
#   .\tools\build.ps1 "models\convnet-sample.onnx" "fp16"

param (
    [String]$inputOnnx,
    [String]$tag
)

$file = $inputOnnx -split '\.'
$file = $file[-1] -split '\\'
$PREFIX = $file[-1]

if ($tag -ne "") {
    $PREFIX = "$PREFIX-$tag"
}

$MODE = "build"
$ONNX_PATH = "models"
$BUILD_PATH = "build"
$ENGINE_PATH = Join-Path $BUILD_PATH "engines"
$LOG_PATH = Join-Path $BUILD_PATH "log\$PREFIX\$MODE"

New-Item -Path $ENGINE_PATH -ItemType Directory -Force
New-Item -Path $LOG_PATH -ItemType Directory -Force

$trtexecPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.1.0.27\bin\trtexec.exe"
$engineFile = Join-Path $ENGINE_PATH "$PREFIX.engine"
$logFile = Join-Path $LOG_PATH "build.log"
$outputLog = Join-Path $LOG_PATH "build_output.log"
$profileLog = Join-Path $LOG_PATH "build_profile.log"
$layerInfoLog = Join-Path $LOG_PATH "build_layer_info.log"

& $trtexecPath --onnx="$inputOnnx" `
              --memPoolSize="workspace:2048" `
              --saveEngine=$engineFile `
              --profilingVerbosity="detailed" `
              --dumpOutput `
              --dumpProfile `
              --dumpLayerInfo `
              --exportOutput=$outputLog `
              --exportProfile=$profileLog `
              --exportLayerInfo=$layerInfoLog `
              --warmUp=200 `
              --iterations=50 `
              --verbose `
              --fp16 | Out-File $logFile
