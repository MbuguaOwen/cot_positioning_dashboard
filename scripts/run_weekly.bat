@echo off
setlocal
cd /d %~dp0\..

python -m cot_bias fetch --combined
python -m cot_bias compute --out outputs
python -m cot_bias dashboard --out outputs

echo Done. Outputs in .\outputs\
endlocal
