let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python310.withPackages (python-pkgs: [
      python-pkgs.torch
      python-pkgs.numpy
      python-pkgs.einops
    ]))
  ];
}
