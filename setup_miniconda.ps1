Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile 'miniconda.exe'
Start-Process -FilePath 'miniconda.exe' -ArgumentList '/S','/D=$env:UserProfile\Miniconda3' -Wait
$env:PATH = "$env:UserProfile\Miniconda3;$env:UserProfile\Miniconda3\Scripts;$env:UserProfile\Miniconda3\Library\bin;$env:PATH"
& $env:UserProfile\Miniconda3\Scripts\activate.ps1
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install -y -c conda-forge sleef