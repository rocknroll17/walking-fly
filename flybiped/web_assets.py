"""Build the browser viewer assets: decimated meshes, model XML, mesh manifest.

Usage: python -m flybiped.web_assets
The 140 MB of original OBJ meshes are simplified (~2.5 MB) so the viewer
loads in seconds; collision and physics are unaffected (the viewer's
physics uses the same XML with the light meshes as visuals).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import fast_simplification
import numpy as np
import trimesh

from flybiped import model as fm

WEB_ASSETS = fm.ROOT / "web/assets"


def main() -> None:
    xml = fm.BIPED_XML if fm.BIPED_XML.exists() else fm.build()
    WEB_ASSETS.mkdir(parents=True, exist_ok=True)
    total = 0
    names = []
    for f in sorted(fm.BUILD_DIR.glob("*.obj")):
        mesh = trimesh.load(f, force="mesh", process=True)
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        v, faces = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32)
        target = min(len(faces), max(600, int(len(faces) * 0.35)), 6000)
        if len(faces) > target:
            v, faces = fast_simplification.simplify(v, faces, target_count=target)
        out = WEB_ASSETS / f.name
        with open(out, "w") as fh:
            fh.write("".join(f"v {x:.5g} {y:.5g} {z:.5g}\n" for x, y, z in v))
            fh.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces))
        total += out.stat().st_size
        names.append(f.name)
    shutil.copy(xml, WEB_ASSETS / "biped.xml")
    (WEB_ASSETS / "manifest.txt").write_text("\n".join(names) + "\n")
    print(f"web assets: {len(names)} meshes, {total / 1e6:.1f} MB -> {WEB_ASSETS}")


if __name__ == "__main__":
    main()
