$ErrorActionPreference = 'Stop'
$python = 'E:\add1\envs\TCFormer1\python.exe'
$script = Join-Path $PSScriptRoot 'run_seed0_inference.py'
$project = Join-Path (Split-Path $PSScriptRoot -Parent) ([string][char]0x84B8 + [char]0x998F + '\TCFormer')

# Keep CPU preprocessing separate from CUDA initialization on this 16 GB machine.
& $python -u $script --project $project --preprocess-only
if ($LASTEXITCODE -ne 0) { throw 'Preprocessing failed' }

foreach ($dataset in @('bcic2a', 'bcic2b')) {
    foreach ($subject in 1..9) {
        & $python -u $script --project $project --datasets $dataset --subjects $subject --precision 32 --quiet-summary
        if ($LASTEXITCODE -ne 0) { throw "Inference failed: $dataset subject $subject" }
    }
}
& $python -u (Join-Path $PSScriptRoot 'verify_results.py')
if ($LASTEXITCODE -ne 0) { throw 'Verification failed' }
