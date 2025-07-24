# Installation Guide - Qualcomm NPU SDK Setup

This guide explains how to set up the Qualcomm NPU SDK for building and running NPU-optimized GPT-2 inference on Copilot+ PCs with Snapdragon X Elite processors.

## Prerequisites

### Hardware Requirements
- **Copilot+ PC** with Qualcomm Snapdragon X Elite processor
- **NPU Support**: Hexagon NPU with QNN SDK compatibility
- **Memory**: Minimum 8GB RAM (16GB recommended for large models)
- **Storage**: 5GB free space for SDK and model weights

### Software Requirements
- **Windows 11 ARM64** (Build 22631 or later)
- **Visual Studio 2022** with ARM64 development tools
- **CMake 3.20** or later
- **Git** for repository management

## Qualcomm NPU SDK Installation

### Step 1: Download Qualcomm AI Stack

1. **Visit Qualcomm Developer Portal:**
   ```
   https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk
   ```

2. **Download Latest QNN SDK:**
   - Select **QNN SDK v2.27.0** or later
   - Choose **Windows ARM64** platform
   - Download the installer package (typically `qairt-windows-x.x.x.zip`)

3. **Install Location:**
   ```
   Default: C:\Qualcomm\AIStack\qairt\2.27.0.240926\
   Custom:  [Your preferred path - note this for later]
   ```

### Step 2: Set Up Environment Variables

Open PowerShell as Administrator and set the following environment variables:

```powershell
# Set QNN SDK path (adjust version as needed)
$QNN_SDK_ROOT = "C:\Qualcomm\AIStack\qairt\2.27.0.240926"
[Environment]::SetEnvironmentVariable("QNN_SDK_ROOT", $QNN_SDK_ROOT, "Machine")

# Add QNN binaries to PATH
$QNN_BIN_PATH = "$QNN_SDK_ROOT\bin\aarch64-windows-msvc"
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
if ($currentPath -notlike "*$QNN_BIN_PATH*") {
    [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$QNN_BIN_PATH", "Machine")
}

# Restart PowerShell to apply changes
```

### Step 3: Copy Required Files to npugpt Repository

**Automatic Setup Script (Recommended):**

Create and run this PowerShell script in the npugpt directory:

```powershell
# setup_qnn.ps1
param(
    [Parameter(Mandatory=$true)]
    [string]$QnnSdkPath
)

$npugptRoot = Get-Location
$qualcommDir = "$npugptRoot\third_party\qualcomm"

Write-Host "Setting up Qualcomm NPU SDK files..."
Write-Host "Source: $QnnSdkPath"
Write-Host "Target: $qualcommDir"

# Create directories
New-Item -ItemType Directory -Force -Path "$qualcommDir\lib"
New-Item -ItemType Directory -Force -Path "$qualcommDir\include"
New-Item -ItemType Directory -Force -Path "$qualcommDir\bin"

# Copy required libraries
$libSource = "$QnnSdkPath\lib\aarch64-windows-msvc"
$libFiles = @(
    "QnnHtp.dll",
    "QnnHtpV73Stub.dll", 
    "QnnSystem.dll",
    "QnnCpu.dll",
    "QnnSaver.dll",
    "libQnnHtpNetRunExtensions.so"
)

foreach ($file in $libFiles) {
    $sourcePath = "$libSource\$file"
    if (Test-Path $sourcePath) {
        Copy-Item $sourcePath "$qualcommDir\lib\" -Force
        Write-Host "✓ Copied $file"
    } else {
        Write-Warning "⚠ Missing $file at $sourcePath"
    }
}

# Copy headers
$includeSource = "$QnnSdkPath\include\QNN"
$includeFiles = @(
    "QnnBackend.h",
    "QnnCommon.h", 
    "QnnContext.h",
    "QnnGraph.h",
    "QnnInterface.h",
    "QnnLog.h",
    "QnnProperty.h",
    "QnnTensor.h",
    "QnnTypes.h",
    "System\QnnSystemInterface.h",
    "HTP\QnnHtpDevice.h",
    "HTP\QnnHtpGraph.h"
)

foreach ($file in $includeFiles) {
    $sourcePath = "$includeSource\$file"
    $targetPath = "$qualcommDir\include\QNN\$file"
    
    # Create subdirectories if needed
    $targetDir = Split-Path $targetPath -Parent
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    
    if (Test-Path $sourcePath) {
        Copy-Item $sourcePath $targetPath -Force
        Write-Host "✓ Copied header $file"
    } else {
        Write-Warning "⚠ Missing header $file at $sourcePath"
    }
}

# Copy essential binaries
$binSource = "$QnnSdkPath\bin\aarch64-windows-msvc"
$binFiles = @(
    "qnn-net-run.exe",
    "qnn-context-binary-generator.exe",
    "qnn-platform-validator.exe"
)

foreach ($file in $binFiles) {
    $sourcePath = "$binSource\$file"
    if (Test-Path $sourcePath) {
        Copy-Item $sourcePath "$qualcommDir\bin\" -Force
        Write-Host "✓ Copied binary $file"
    } else {
        Write-Warning "⚠ Missing binary $file"
    }
}

Write-Host "
✅ Qualcomm NPU SDK setup complete!

Next steps:
1. Verify installation: .\third_party\qualcomm\bin\qnn-platform-validator.exe
2. Build the project: cmake --build build --config Release
3. Run demo: .\build\Release\npu_gpt2_demo.exe
"
```

**Run the setup script:**
```powershell
# In npugpt directory
.\setup_qnn.ps1 -QnnSdkPath "C:\Qualcomm\AIStack\qairt\2.27.0.240926"
```

**Manual Setup (Alternative):**

If the script fails, manually copy these files:

1. **Libraries (`third_party/qualcomm/lib/`):**
   ```
   From: C:\Qualcomm\AIStack\qairt\2.27.0.240926\lib\aarch64-windows-msvc\
   Copy: QnnHtp.dll, QnnHtpV73Stub.dll, QnnSystem.dll, QnnCpu.dll
   ```

2. **Headers (`third_party/qualcomm/include/QNN/`):**
   ```
   From: C:\Qualcomm\AIStack\qairt\2.27.0.240926\include\QNN\
   Copy all .h files maintaining directory structure
   ```

### Step 4: Verify Installation

**Check NPU Hardware:**
```powershell
# Run NPU platform validator
.\third_party\qualcomm\bin\qnn-platform-validator.exe

# Expected output should show:
# ✓ Hexagon NPU detected
# ✓ QNN HTP backend available
# ✓ NPU driver version compatible
```

**Test Build System:**
```powershell
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -G "Visual Studio 17 2022" -A ARM64

# Build (should complete without QNN-related errors)
cmake --build . --config Release
```

## Troubleshooting

### Common Issues

#### Issue 1: "QnnHtp.dll not found"
**Solution:**
```powershell
# Check if DLL is in the right location
Get-ChildItem -Recurse -Name "QnnHtp.dll" .\third_party\qualcomm\

# If missing, re-run setup script or copy manually
# Ensure the DLL is in third_party/qualcomm/lib/
```

#### Issue 2: "NPU device not detected"
**Solution:**
1. **Update drivers:**
   ```
   Windows Update → Check for updates
   Device Manager → Neural Processors → Update driver
   ```

2. **Verify hardware:**
   ```powershell
   # Check device manager for NPU
   Get-PnpDevice -FriendlyName "*Neural*" -Status OK
   ```

3. **Enable NPU in BIOS/UEFI** (if available)

#### Issue 3: "CMake cannot find QNN headers"
**Solution:**
```powershell
# Verify header structure
Get-ChildItem -Recurse .\third_party\qualcomm\include\

# Should show QNN/ subdirectory with .h files
# If missing, re-run header copy step
```

#### Issue 4: Build errors with Visual Studio
**Solution:**
1. **Install required workloads:**
   ```
   Visual Studio Installer → Modify → Workloads:
   ✓ Desktop development with C++
   ✓ Game development with C++ (for ARM64 tools)
   ```

2. **ARM64 build tools:**
   ```
   Individual Components:
   ✓ MSVC v143 - VS 2022 C++ ARM64 build tools
   ✓ Windows 11 SDK (10.0.22621.0)
   ```

### Performance Validation

After successful installation, run performance validation:

```powershell
# Build and run basic test
cd build\Release
.\npu_gpt2_demo.exe

# Expected output:
# 🚀 NPU GPT-2 Demo
# ✅ NPU context initialized (150ms)
# ✅ Model loaded successfully  
# ✅ Generation completed: "Hello world" → "Hello world, this is a test..."
# 📊 Performance: 45.2ms per token
```

### Support and Resources

- **Qualcomm Developer Documentation:** https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/introduction.html
- **QNN SDK Reference:** Available in SDK installation directory
- **NPU Programming Guide:** `[SDK_PATH]\docs\QNN_programming_guide.pdf`
- **Hardware Specifications:** Snapdragon X Elite NPU specifications

### Advanced Configuration

#### Custom SDK Location

If you installed the QNN SDK in a custom location, update CMakeLists.txt:

```cmake
# Set custom QNN SDK path
set(QNN_SDK_ROOT "D:/CustomPath/QNN_SDK" CACHE PATH "QNN SDK Root Directory")
```

#### Debug vs Release Builds

```powershell
# Debug build (slower, with debugging symbols)
cmake --build build --config Debug

# Release build (optimized performance)  
cmake --build build --config Release
```

#### Enable Detailed NPU Logging

```powershell
# Set environment variable for verbose NPU logging
$env:QNN_LOG_LEVEL = "DEBUG"
.\build\Release\npu_gpt2_demo.exe
```

---

✅ **Installation Complete!** You're now ready to build and run NPU-optimized GPT-2 inference on your Snapdragon X Elite system.