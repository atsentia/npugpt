@echo off
cd /d C:\Users\atvei\npullm\npugpt
if not exist build\Release\Release mkdir build\Release\Release
cd build\Release\Release
cl.exe /O2 /EHsc /std:c++17 ..\..\..\simple_kvcache_demo.cpp /Fe:simple_kvcache_demo.exe
if %errorlevel% equ 0 (
    echo Build successful!
    simple_kvcache_demo.exe
) else (
    echo Build failed!
)