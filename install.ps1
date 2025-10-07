$RELEASE_URL = "https://raw.githubusercontent.com/LAAC-LSCP/ChildProject/docs/rework_installation/env_windows.yml"

curl.exe -o $env:TEMP\env.yml $RELEASE_URL

if (Get-Command micromamba) {
    $ENV_MANAGER = "micromamba"
} elseif (Get-Command conda) {
    $ENV_MANAGER = "conda"
} else {
    echo "not found"
    exit 1
}

$success = "Installation Completed for environment childproject, activate it with: \n$ENV_MANAGER activate childproject"
# $install = Invoke-Expression ("($ENV_MANAGER env create -f $env:TEMP\env.yml) -and (echo $SUCCESS)")
micromamba env create -f $env:TEMP\env.yml

if($?)
{
   "Installing git-annex"
   Invoke-Expression ("$ENV_MANAGER activate childproject")
   uv tool install git-annex
}
else
{
   "Something went wrong in the environment creation"
}

