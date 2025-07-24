#!/usr/bin/env python3
"""
NPU GPT-2 Dependencies Installer
Copies QNN SDK headers and libraries to the build directory for compilation.
"""

import os
import sys
import shutil
from pathlib import Path

def install_qnn_dependencies(qnn_sdk_path, build_output_dir):
    """Install QNN SDK dependencies for npugpt build"""
    
    qnn_path = Path(qnn_sdk_path)
    output_path = Path(build_output_dir)
    
    if not qnn_path.exists():
        print(f"ERROR: QNN SDK not found at: {qnn_path}")
        return False
    
    print(f"Installing QNN SDK dependencies...")
    print(f"   Source: {qnn_path}")
    print(f"   Target: {output_path}")
    
    # Create third_party/qualcomm directory structure
    qualcomm_dir = Path("third_party/qualcomm")
    qualcomm_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy QNN headers
    qnn_include_src = qnn_path / "include"
    qnn_include_dst = qualcomm_dir / "include"
    
    if qnn_include_src.exists():
        if qnn_include_dst.exists():
            shutil.rmtree(qnn_include_dst)
        shutil.copytree(qnn_include_src, qnn_include_dst)
        print(f"SUCCESS: Copied QNN headers to {qnn_include_dst}")
    else:
        print(f"WARNING: QNN headers not found at {qnn_include_src}")
    
    # Copy QNN libraries
    qnn_lib_src = qnn_path / "lib" / "aarch64-windows-msvc"
    qnn_lib_dst = qualcomm_dir / "lib"
    
    if qnn_lib_src.exists():
        qnn_lib_dst.mkdir(exist_ok=True)
        
        # Copy essential QNN DLLs
        qnn_dlls = [
            "QnnHtp.dll",
            "QnnHtpV73Stub.dll", 
            "QnnSystem.dll",
            "QnnCpu.dll"
        ]
        
        for dll in qnn_dlls:
            src_dll = qnn_lib_src / dll
            dst_dll = qnn_lib_dst / dll
            
            if src_dll.exists():
                shutil.copy2(src_dll, dst_dll)
                print(f"SUCCESS: Copied {dll}")
            else:
                print(f"WARNING: {dll} not found at {src_dll}")
        
        # Also copy to build output directory for runtime
        output_path.mkdir(parents=True, exist_ok=True)
        for dll in qnn_dlls:
            src_dll = qnn_lib_dst / dll
            dst_dll = output_path / dll
            
            if src_dll.exists():
                shutil.copy2(src_dll, dst_dll)
                print(f"SUCCESS: Copied {dll} to build output")
                
    else:
        print(f"WARNING: QNN libraries not found at {qnn_lib_src}")
    
    print(f"SUCCESS: QNN SDK dependencies installed successfully!")
    return True

def main():
    if len(sys.argv) != 3:
        print("Usage: python install_deps.py <qnn_sdk_path> <build_output_dir>")
        print("Example: python install_deps.py \"C:\\Qualcomm\\AIStack\\qairt\\2.27.0.240926\" \".\\build\\Release\"")
        sys.exit(1)
    
    qnn_sdk_path = sys.argv[1]
    build_output_dir = sys.argv[2]
    
    success = install_qnn_dependencies(qnn_sdk_path, build_output_dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()