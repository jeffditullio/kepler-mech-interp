"""Build papers/ditullio-e-register/paper/draft.pdf from draft.md.

Pipeline: draft.md -> strip HTML working comments -> pandoc (markdown -> typst)
-> typst compile -> draft.pdf. Pandoc and typst both ship as Python wheels
(pypandoc-binary, typst in the dev dependency group) -- no system TeX needed.
The HTML comments in draft.md are working notes and never reach the PDF.

Run:  uv run python papers/ditullio-e-register/repro/build_pdf.py
"""

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

import pypandoc
import typst

PAPER_DIR = Path(__file__).resolve().parent.parent / "paper"

AUTHOR_LINE = "Jeff DiTullio · jeffditullio@gmail.com"

# Page + text defaults, and a global image rule so figures fit the text width.
TYPST_PREAMBLE = """\
#set page(paper: "us-letter", margin: (x: 1.5in, y: 1in), numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set image(width: 100%)
#show heading: set text(hyphenate: false)
#show table.cell.where(y: 0): strong
#show heading.where(level: 1): set text(size: 16pt)
#show heading.where(level: 2): set text(size: 13pt)
#show heading.where(level: 3): set text(size: 11pt)
#show link: set text(fill: blue)

"""


def check_caption_numbers(markdown: str) -> None:
    """Renumbering guard: figure and table caption numbers must each run 1..N
    in order, and no in-text reference may point past the last caption.
    Validation only -- draft.md stays the single source of the numbering."""
    for label, caption_pattern, ref_pattern in (
        ("Fig.", r"^\*\*Fig\. (\d+)\.", r"Fig(?:ure)?\.?\s+(\d+)"),
        ("Table", r"^\*\*Table (\d+)\.", r"Table\s+(\d+)"),
    ):
        captions = [int(k) for k in re.findall(caption_pattern, markdown, flags=re.MULTILINE)]
        if captions != list(range(1, len(captions) + 1)):
            raise SystemExit(f"{label} captions out of order or gapped: {captions}")
        refs = [int(k) for k in re.findall(ref_pattern, markdown)]
        stale = sorted({k for k in refs if k > len(captions)})
        if stale:
            raise SystemExit(f"in-text refs past the last caption ({label} {len(captions)}): {stale}")


def main() -> None:
    markdown = (PAPER_DIR / "draft.md").read_text()
    # <!-- width=N% --> after an image is the draft's width notation: GitHub
    # hides it (a bare {width=N%} attribute would render as literal text), so
    # translate it back into the pandoc attribute BEFORE comments are stripped,
    # then fail loudly on any width comment the translation missed.
    markdown = re.sub(r"\)\s*<!-- width=(\d+%) -->", r"){width=\1}", markdown)
    if re.search(r"<!--\s*width=", markdown):
        raise SystemExit("malformed width comment; expected ')<!-- width=N% -->' right after an image")
    markdown = re.sub(r"<!--.*?-->", "", markdown, flags=re.DOTALL)
    # <br> is the draft's in-cell line-break notation: GitHub renders it
    # natively, and pandoc would silently DROP it on the way to typst, so
    # translate it to a raw typst linebreak here.
    markdown = markdown.replace("<br>", "`#linebreak()`{=typst}")
    check_caption_numbers(markdown)

    # Draft stamp under the author line: build date + content hash of the
    # comment-stripped source, so two PDFs differ iff their stamps differ
    # (uncommitted edits included; comment-only edits excluded). Remove for
    # the arXiv submission -- arXiv stamps its own date/version.
    content_hash = hashlib.sha256(markdown.encode()).hexdigest()[:7]
    build_date = datetime.now(tz=UTC).astimezone().date()
    stamp = f"*Draft — {build_date.isoformat()} · {content_hash}*"
    if AUTHOR_LINE not in markdown:
        raise SystemExit(f"author line not found in draft.md; can't place stamp: {AUTHOR_LINE!r}")
    markdown = markdown.replace(AUTHOR_LINE, f"{AUTHOR_LINE}\n\n{stamp}", 1)
    # Drop image alt texts: pandoc would otherwise promote each image to an
    # auto-captioned "Figure N" that duplicates the draft's own Fig. captions.
    markdown = re.sub(r"!\[[^\]]*\]\(", "![](", markdown)

    # -citations: the draft has no pandoc-style cite keys, and stray "@800k"
    # text would otherwise be parsed as one (breaking the typst compile).
    typst_source = pypandoc.convert_text(markdown, to="typst", format="markdown-citations")

    # Every image comes out of pandoc as an inline #box(image(...)) -- the
    # alt-strip above leaves them captionless, so they never become #figure --
    # and inline boxes sit left-aligned. Center them.
    typst_source = re.sub(r"#box\(image\((.*?)\)\)", r"#align(center, image(\1))", typst_source)

    # Captions (paragraphs opening with bold "Fig. N." or "Table N."): 9pt, so
    # they read as captions rather than body text, and pulled tight toward what
    # they caption -- figures sit under their image, tables over their table.
    # Table captions are sticky so a page break cannot orphan them from the table.
    def style_caption(match: re.Match[str]) -> str:
        caption = match.group(1)
        pull = "above: 0.5em" if caption.startswith("#strong[Fig.") else "below: 0.5em, sticky: true"
        return f"\n#block({pull})[#text(size: 9pt)[{caption}]]\n\n"

    typst_source = re.sub(
        r"\n(#strong\[(?:Fig\.|Table) .*?)\n\n",
        style_caption,
        typst_source,
        flags=re.DOTALL,
    )
    # typst.compile needs a real file under root for relative paths; use a
    # transient one so no draft.typ intermediate lingers -- draft.md is the
    # ONLY source.
    typst_path = PAPER_DIR / ".draft.typ.tmp"
    typst_path.write_text(TYPST_PREAMBLE + typst_source)
    try:
        pdf_path = PAPER_DIR / "draft.pdf"
        # root = papers/ditullio-e-register/ so the draft's ../figures/*.png image paths resolve.
        typst.compile(str(typst_path), output=str(pdf_path), root=str(PAPER_DIR.parent))
    finally:
        typst_path.unlink(missing_ok=True)
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
