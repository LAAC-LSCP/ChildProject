# check if VERSION env variable is set, otherwise use "latest"
$RELEASE_URL = "https://raw.githubusercontent.com/LAAC-LSCP/ChildProject/docs/rework_installation/env_windows.yml"

curl.exe -L -o %TEMP%\env.yml $RELEASE_URL

if (Get-Command git-annex) {
    echo "Git annex found"
} else {
    echo "Git annex not found, installing it..."
    curl.exe -L -o %TEMP%\git-annex-installer.exe https://downloads.kitenet.net/git-annex/windows/current/git-annex-installer.exe
    %TEMP%\git-annex-installer.exe
}

if (Get-Command micromamba) {
    $ENV_MANAGER = "micromamba"
} elseif (Get-Command conda) {
    $ENV_MANAGER = "conda"
} else {
    echo "not found"
    exit 1
}

$success = "Installation Completed for environment childproject, activate it with: \n$ENV_MANAGER activate childproject"
Invoke-Expression ("($ENV_MANAGER env create -f $env:TEMP\env.yml) -and (echo $SUCCESS)")

