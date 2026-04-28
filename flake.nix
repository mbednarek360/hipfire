{
  description = "LLM inference for AMD RDNA GPUs — Rust + HIP";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.rocmPackages = {
            accept_license = true;
          };
        };

        rocm = pkgs.rocmPackages;
        clr = rocm.clr;

        # ─── Shell scripts as Nix derivations ─────────────────────
        # Using pkgs.writeShellScriptBin with ''...'' (raw strings)
        # avoids all ${} escaping issues — bash variables stay literal.

       # Absolute path to the CLI directory (known at build time).
        hipfireCliDir = "${hipfire-bin}/share/hipfire-cli";

        # The hipfire CLI wrapper script with hardcoded bun and CLI dir paths.
        hipfireCliWrapper = pkgs.writeTextFile {
          name = "hipfire";
          executable = true;
          destination = "/bin/hipfire";
          text = ''
            #!/bin/bash
            set +u
            set -e
            BUN="${pkgs.bun}/bin/bun"
            if [ ! -x "$BUN" ]; then
              echo "hipfire: bun not found at $BUN" >&2
              exit 127
            fi
            CLI_DIR="${hipfireCliDir}"
            if [ ! -f "$CLI_DIR/index.ts" ]; then
              echo "hipfire: CLI not found at $CLI_DIR" >&2
              exit 127
            fi
            exec "$BUN" run "$CLI_DIR/index.ts" "$@"
          '';
        };

        # Kernel pre-compilation helper.
        # Searches for pre-built .hsaco blobs and copies them to
        # ~/.hipfire/bin/kernels/<arch>/ for the daemon to load.
        hipfireKernels = pkgs.writeShellApplication {
          name = "hipfire-kernels";
          runtimeInputs = [ pkgs.gnugrep ];
         text = ''
            set +u
            set -e
            export HIPFIRE_DIR="$HOME/.hipfire"
            export HIPFIRE_BIN="${placeholder "out"}/bin"
            KERNEL_LIST=""

            # Collect available pre-compiled kernel sets from known paths.
            for dir in \
              "$HIPFIRE_DIR/bin/kernels/compiled" \
              "$HIPFIRE_BIN/kernels/compiled"; do
              if [ -d "$dir" ]; then
                for arch_dir in "$dir"/*/; do
                  arch=$(basename "$arch_dir")
                  KERNEL_LIST="$KERNEL_LIST $arch"
                done
              fi
            done

            # --list mode: print available architectures.
            if [ "$1" = "--list" ]; then
              for a in $KERNEL_LIST; do echo "$a"; done
              exit 0
            fi

            GPU_ARCH="$1"
            if [ -z "$GPU_ARCH" ]; then
              # Auto-detect via kfd topology (same logic as install.sh).
              for node_props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
                [ -f "$node_props" ] || continue
                ver=$(grep -oP 'gfx_target_version\s+\K\d+' "$node_props" 2>/dev/null || true)
                case "$ver" in
                  90008)          GPU_ARCH="gfx908";  break ;;
                  100100)         GPU_ARCH="gfx1010"; break ;;
                  100300|100302)  GPU_ARCH="gfx1030"; break ;;
                  110000|110001)  GPU_ARCH="gfx1100"; break ;;
                  110501)         GPU_ARCH="gfx1151"; break ;;
                  120000)         GPU_ARCH="gfx1200"; break ;;
                  120001)         GPU_ARCH="gfx1201"; break ;;
                esac
              done
              # Fallback: use first available.
              if [ -z "$GPU_ARCH" ]; then
                GPU_ARCH=$(echo "$KERNEL_LIST" | tr ' ' '\n' | head -1)
              fi
            fi

            if [ -z "$GPU_ARCH" ]; then
              echo "ERROR: no GPU arch detected and no pre-compiled kernels found." >&2
              exit 1
            fi

            KERNEL_DEST="$HIPFIRE_DIR/bin/kernels/compiled/$GPU_ARCH"
            mkdir -p "$KERNEL_DEST"

            # Copy from any known source path that has this arch.
            for src_dir in \
              "$HIPFIRE_DIR/bin/kernels/compiled/$GPU_ARCH" \
              "$HIPFIRE_BIN/kernels/compiled/$GPU_ARCH"; do
              if [ -d "$src_dir" ]; then
                cp "$src_dir"/*.hsaco "$KERNEL_DEST/" 2>/dev/null || true
                cp "$src_dir"/*.hash "$KERNEL_DEST/" 2>/dev/null || true
                echo "Copied kernels for $GPU_ARCH from $src_dir"
                break
              fi
            done

            echo "Kernels installed to $KERNEL_DEST/"
          '';
        };

        # The main hipfire Rust workspace build.
        # Builds the full workspace (all examples) since buildRustWorkspace
        # is not available in this nixpkgs version.
        hipfire-bin = pkgs.rustPlatform.buildRustPackage {
          pname = "hipfire";
          version = "0.1.8-alpha.2";
          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          # ROCm headers needed for kernel compilation during build.
          buildInputs = [
            rocm.rocm-core
            rocm.rocm-device-libs
          ];

          # Pass ROCm paths so hipcc is found during kernel JIT / precompile.
          nativeBuildInputs = [
            pkgs.makeWrapper
            rocm.rocm-core
          ];

          doCheck = false;

          buildPhase = ''
            cargoBuildHook
            # Also build all examples with deltanet feature.
            cargo build --release --features deltanet --example daemon \
              --example dflash_spec_demo --example encode_prompt \
              --example run --example infer --example infer_hfq \
              --example test_kernels \
              -p engine 2>/dev/null || true
            # Build hipfire-quantize crate.
            cargo build --release -p hipfire-quantize 2>/dev/null || true
          '';

          installPhase = ''
              mkdir -p $out/bin
              # Copy all engine examples (only actual binaries, not .d/.o/hash files).
              for f in target/release/examples/*; do
                [ -f "$f" ] || continue
                case "$f" in
                  *.d|*.o) continue ;;
                esac
                basename_f=$(basename "$f")
                # Skip cargo fingerprinted copies (e.g., daemon-f9993d38fd088eb6).
                case "$basename_f" in
                  *-*[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) continue ;;
                esac
                cp "$f" $out/bin/
              done
              # Copy hipfire-quantize binary.
              cp target/release/hipfire-quantize $out/bin/ 2>/dev/null || true

              # Bundle the CLI TypeScript app.
              mkdir -p $out/share/hipfire-cli
              cp -r cli/* $out/share/hipfire-cli/

              # Install the kernel helper.
              cp ${hipfireKernels}/bin/hipfire-kernels $out/bin/hipfire-kernels
          '';

          meta = with pkgs.lib; {
            description = "LLM inference engine for AMD RDNA GPUs";
            homepage = "https://github.com/Kaden-Schutt/hipfire";
            license = licenses.mit;
            platforms = platforms.linux;
            maintainers = [ ];
          };
        };

        # Self-contained package with ROCm-wrapped binaries + CLI bundled.
        hipfire = pkgs.stdenvNoCC.mkDerivation {
          name = "hipfire";
          src = hipfire-bin;

          nativeBuildInputs = [ pkgs.makeWrapper pkgs.bun ];

installPhase = ''
            mkdir -p $out/bin $out/share
            cp -r ${hipfire-bin}/bin/* $out/bin/
            cp -r ${hipfire-bin}/share/* $out/share/

            # Install the hipfire CLI wrapper with hardcoded CLI dir path.
            cp ${hipfireCliWrapper}/bin/hipfire $out/bin/hipfire

            # Wrap every binary to ensure ROCm libs are discoverable.
            for bin in $out/bin/*; do
              [ -f "$bin" ] || continue
              [ -x "$bin" ] || continue
              # Skip directories (kernels/compiled) and scripts that already have shebangs.
              case "$(file -b "$bin")" in
                *script*text*) continue ;;
              esac
              wrapProgram "$bin" \
               --prefix LD_LIBRARY_PATH : "${rocm.rocm-runtime}/lib:${rocm.rocm-core}/lib" \
               --prefix PATH : "${rocm.rocm-core}/bin" \
               --set HIP_PATH "${clr}" \
               --set HIP_PLATFORM amd \
               --set ROCM_PATH "${clr}" \
               --set HSA_PATH "${clr}"
            done
          '';
        };

        # Development shell with all build dependencies.
        devShell = pkgs.mkShell {
          name = "hipfire-dev";
          packages = [
            # Rust toolchain.
            pkgs.rustup
            pkgs.cargo-expand
            pkgs.rustfmt
            pkgs.clippy

            # ROCm / HIP build tools.
            clr
            rocm.rocm-core
            rocm.rocm-device-libs
            rocm.rocm-runtime
            pkgs.llvmPackages.bintools

            # CLI tooling.
            pkgs.bun

            # Kernel pre-compilation.
            pkgs.gnused
            pkgs.gnugrep
            pkgs.file

            # Utilities.
            pkgs.curl
            pkgs.gitMinimal
          ];

          shellHook = ''
            export HIP_PATH="${clr}"
            export HIP_PLATFORM=amd
            export ROCM_PATH="${clr}"
            export HSA_PATH="${clr}"
            export HIP_CLANG_PATH="${rocm.llvm.clang-unwrapped}/bin"
            export DEVICE_LIB_PATH="${rocm.rocm-device-libs}/amdgcn/bitcode"
            export LD_LIBRARY_PATH="${rocm.rocm-runtime}/lib:${rocm.rocm-core}/lib:$LD_LIBRARY_PATH"

            echo "=== hipfire dev shell ==="
            echo "  ROCm: ${clr}"
            echo "  hipcc: ${clr}/bin/hipcc"
            echo "  bun:   ${pkgs.bun}/bin/bun"
            echo ""
            echo "Build: cargo build --release --features deltanet -p engine"
            echo "Run:   ./target/release/examples/daemon"
          '';
        };

      in
      {
        packages = rec {
          default = hipfire;
          inherit hipfire hipfire-bin;
        };

        apps.default = flake-utils.lib.mkApp {
          drv = hipfire;
        };

        devShells.default = devShell;
      }
    );
}    
