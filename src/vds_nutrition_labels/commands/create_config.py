
import os
import pathlib


def _path_to_template() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "config" / "template" / "vds-config-template.toml"

def create_config_template(target: pathlib.Path | None = None) -> pathlib.Path:
    template_path = _path_to_template()
    if not template_path.exists():
        raise FileNotFoundError(
            f"Built-in config template not found: {template_path}")
        
    if target is None:
        target = pathlib.Path.cwd() / template_path.name
    if os.path.isdir(target):
        target = target / "vds-config.toml"
    if target.exists():
        raise FileExistsError(f"Config file already exists: {target}")

    target.write_bytes(template_path.read_bytes())
    return target
