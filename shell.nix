{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python310Packages.pandas
    pkgs.python310Packages.matplotlib
    pkgs.python310Packages.opencv4
  ];
}
