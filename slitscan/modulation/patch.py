"""Parse --mod strings and YAML patch files."""

from dataclasses import dataclass


@dataclass
class ModEntry:
    dest: str         # "vanguard" | "max_delay" | "slice_width" | "fill_alpha"
    osc: str          # "sine" | "triangle"
    rate_str: str     # unparsed: "0.5cyc", "2hz", "4frames"
    depth: float = 1.0
    phase: float = 0.0
    offset: float = 0.0


def parse_mod_string(s: str) -> ModEntry:
    """Parse CLI --mod string: 'dest=osc:rate=...,depth=...,phase=...,offset=...'"""
    try:
        dest_part, rest = s.split("=", 1)
        dest = dest_part.strip()
        osc_part, params_str = rest.split(":", 1)
        osc = osc_part.strip()

        params = {}
        for item in params_str.split(","):
            item = item.strip()
            if "=" in item:
                k, v = item.split("=", 1)
                params[k.strip()] = v.strip()

        return ModEntry(
            dest=dest,
            osc=osc,
            rate_str=params.get("rate", "1cyc"),
            depth=float(params.get("depth", 1.0)),
            phase=float(params.get("phase", 0.0)),
            offset=float(params.get("offset", 0.0)),
        )
    except (ValueError, KeyError) as exc:
        raise ValueError(
            f"Invalid --mod string {s!r}. "
            "Expected format: dest=osc:rate=<r>,depth=<d>[,phase=<p>][,offset=<o>]"
        ) from exc


def load_mod_file(path: str) -> tuple[dict, list[ModEntry]]:
    """Load YAML patch file. Returns (base_params, list[ModEntry])."""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)

    mods = []
    for entry in data.get("mods", []):
        mods.append(ModEntry(
            dest=entry["dest"],
            osc=entry["osc"],
            rate_str=str(entry.get("rate", "1cyc")),
            depth=float(entry.get("depth", 1.0)),
            phase=float(entry.get("phase", 0.0)),
            offset=float(entry.get("offset", 0.0)),
        ))

    base_params = {k: v for k, v in data.items() if k != "mods"}
    return base_params, mods
