"""Render REPORT_ko.md (headings, paragraphs, lists, tables, figures, code) to REPORT_ko.pdf.

Needs reportlab and a Korean TTF font (NanumGothic, SIL OFL, from github.com/google/fonts).
Run:  python diagnostics/report/build_pdf.py --fonts ~/YeahPick/data/fonts
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, Image, KeepTogether, ListFlowable, ListItem, Paragraph,
                                Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
INK, INK2, MUTED, RULE, ZEBRA, ACCENT = (colors.HexColor(c) for c in
                                         ("#111111", "#4a4a46", "#7a7974", "#d9d8d1", "#f4f3ef", "#2a78d6"))


def register_fonts(font_dir: Path) -> None:
    pdfmetrics.registerFont(TTFont("Nanum", str(font_dir / "NanumGothic-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("Nanum-Bold", str(font_dir / "NanumGothic-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("NanumMono", str(font_dir / "NanumGothicCoding-Regular.ttf")))
    pdfmetrics.registerFontFamily("Nanum", normal="Nanum", bold="Nanum-Bold", italic="Nanum", boldItalic="Nanum-Bold")


def styles() -> dict:
    base = dict(fontName="Nanum", textColor=INK, leading=15, fontSize=10)
    return {
        "title": ParagraphStyle("title", **{**base, "fontName": "Nanum-Bold", "fontSize": 19, "leading": 25, "spaceAfter": 8}),
        "h2": ParagraphStyle("h2", **{**base, "fontName": "Nanum-Bold", "fontSize": 14, "leading": 19, "spaceBefore": 14, "spaceAfter": 6}),
        "h3": ParagraphStyle("h3", **{**base, "fontName": "Nanum-Bold", "fontSize": 11.5, "leading": 16, "spaceBefore": 10, "spaceAfter": 4}),
        "body": ParagraphStyle("body", **{**base, "spaceAfter": 6}),
        "quote": ParagraphStyle("quote", **{**base, "leftIndent": 10, "textColor": INK2, "borderPadding": (4, 6, 4, 8),
                                            "borderColor": ACCENT, "borderWidth": 0, "spaceBefore": 4, "spaceAfter": 8}),
        "caption": ParagraphStyle("caption", **{**base, "fontSize": 8.5, "leading": 12, "textColor": MUTED, "alignment": TA_CENTER, "spaceAfter": 10}),
        "cell": ParagraphStyle("cell", **{**base, "fontSize": 8, "leading": 10.5}),
        "cellhead": ParagraphStyle("cellhead", **{**base, "fontName": "Nanum-Bold", "fontSize": 8, "leading": 10.5}),
        "code": ParagraphStyle("code", fontName="NanumMono", fontSize=8, leading=11, textColor=INK, backColor=ZEBRA,
                               borderPadding=6, leftIndent=4, spaceBefore=4, spaceAfter=10),
    }


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r'<font name="NanumMono">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def table_flowable(rows: list[list[str]], st: dict, width: float) -> Table:
    ncol = len(rows[0])
    # column widths proportional to content length, with a floor so short columns stay readable
    # display width: CJK characters take about twice the width of Latin ones
    dw = lambda t: sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in t)
    lens = [max(dw(r[c]) if c < len(r) else 0 for r in rows) for c in range(ncol)]
    heads = [dw(rows[0][c]) for c in range(ncol)]
    weights = [min(max(l, h_, 11), 40) for l, h_ in zip(lens, heads)]
    total = sum(weights)
    widths = [width * w / total for w in weights]
    data = [[Paragraph(inline(cell), st["cellhead"] if i == 0 else st["cell"]) for cell in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2), ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
             ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    style += [("BACKGROUND", (0, i), (-1, i), ZEBRA) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return t


def image_flowable(path: Path, width: float, max_h: float) -> Image:
    iw, ih = ImageReader(str(path)).getSize()
    scale = min(width / iw, max_h / ih)
    return Image(str(path), width=iw * scale, height=ih * scale)


def build(md_path: Path, pdf_path: Path) -> None:
    st = styles()
    title = next((l[2:].strip() for l in md_path.read_text().splitlines() if l.startswith("# ")), md_path.stem)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title=title,
                            author="예픽")
    width = doc.width
    lines = md_path.read_text().splitlines()
    story, i, para = [], 0, []

    def flush():
        if para:
            story.append(Paragraph("<br/>".join(inline(p) for p in para), st["body"]))
            para.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            flush(); i += 1; continue
        if s.startswith("```"):
            flush(); code = []; i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i += 1
            story.append(Preformatted("\n".join(code), st["code"], maxLineLength=92, newLineChars="  ")); i += 1; continue
        if s.startswith("# "):
            flush(); story.append(Paragraph(inline(s[2:]), st["title"])); i += 1; continue
        if s.startswith("## "):
            flush(); story.append(Paragraph(inline(s[3:]), st["h2"])); i += 1; continue
        if s.startswith("### "):
            flush(); story.append(Paragraph(inline(s[4:]), st["h3"])); i += 1; continue
        if s == "---":
            flush(); story.append(HRFlowable(width="100%", thickness=0.5, color=RULE, spaceBefore=4, spaceAfter=8)); i += 1; continue
        if s.startswith(">"):
            flush(); story.append(Paragraph(inline(s.lstrip("> ")), st["quote"])); i += 1; continue
        m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", s)
        if m:
            flush()
            img = image_flowable(md_path.parent / m.group(1), width, 120 * mm)
            block = [Spacer(1, 4), img]
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("*"):
                block.append(Paragraph(inline(lines[i + 1].strip()), st["caption"])); i += 1
            story.append(KeepTogether(block)); i += 1; continue
        if s.startswith("|"):
            flush(); rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            story += [table_flowable(rows, st, width), Spacer(1, 8)]; continue
        if re.match(r"^(- |\d+\. )", s):
            flush(); items, numbered = [], bool(re.match(r"^\d+\. ", s))
            while i < len(lines) and re.match(r"^(- |\d+\. )", lines[i].strip()):
                txt = re.sub(r"^(- |\d+\. )", "", lines[i].strip())
                items.append(ListItem(Paragraph(inline(txt), st["body"]), leftIndent=14))
                i += 1
            story.append(ListFlowable(items, bulletType="1" if numbered else "bullet", start="1" if numbered else None,
                                      bulletFontName="Nanum", bulletFontSize=9, leftIndent=14))
            continue
        para.append(s); i += 1
    flush()

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Nanum", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc_.leftMargin, 9 * mm, f"예픽 · {title}")
        canvas.drawRightString(A4[0] - doc_.rightMargin, 9 * mm, f"{doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fonts", type=Path, required=True)
    ap.add_argument("--md", type=Path, default=HERE / "REPORT_ko.md")
    ap.add_argument("--out", type=Path, default=HERE / "REPORT_ko.pdf")
    args = ap.parse_args()
    register_fonts(args.fonts)
    build(args.md, args.out)
    print(args.out)
