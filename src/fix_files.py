import plistlib
from pathlib import Path

docs = Path("docs")
out = Path("docs_clean")
out.mkdir(exist_ok=True)

for f in docs.glob("*.txt"):
    raw = f.read_bytes()
    if raw.startswith(b"bplist00"):
        text = plistlib.loads(raw)["UTI-Data"]["public.utf8-plain-text"]
        kind = "bplist"
    elif raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        if len(raw) % 2:
            raw = raw[:-1]
        text = raw.decode("utf-16")
        kind = "utf-16 (BOM)"
    elif raw[1:2] == b"\x00":
        if len(raw) % 2:
            raw = raw[:-1]
        text = raw.decode("utf-16-le")
        kind = "utf-16-le (no BOM)"
    elif raw[:1] == b"\x00":
        if len(raw) % 2:
            raw = raw[:-1]
        text = raw.decode("utf-16-be")
        kind = "utf-16-be (no BOM)"
    else:
        text = raw.decode("utf-8")
        kind = "utf-8"
    (out / f.name).write_text(text, encoding="utf-8")
    print(f"{f.name}: {kind}, {len(text)} characters")