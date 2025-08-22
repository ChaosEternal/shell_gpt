{
  pkgs ? import <nixpkgs> {} 
}:

pkgs.python3Packages.buildPythonApplication rec {
  pname = "shell-gpt";
  version = "1.4.5-ce";
  pyproject = true;

  src = ./.;

  pythonRelaxDeps = [
    "requests"
    "rich"
    "distro"
    "typer"
    "instructor"
    "jinja2"
  ];

  build-system = with pkgs.python3Packages; [ hatchling ];

  propagatedBuildInputs = with pkgs.python3Packages; [
    jinja2
    requests
    click
    distro
    instructor
    litellm
    openai
    prompt-toolkit
    rich
    typer
  ];

  buildInputs = with pkgs.python3Packages; [
    litellm
  ];

  # Tests want to read the OpenAI API key from stdin
  doCheck = false;

  meta = {
    description = "Access ChatGPT from your terminal";
    homepage = "https://github.com/TheR1D/shell_gpt";
    changelog = "https://github.com/TheR1D/shell_gpt/releases/tag/${version}";
    license = pkgs.lib.licenses.mit;
    mainProgram = "sgpt";
  };
}
