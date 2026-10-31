# -*- coding: utf-8 -*-
"""Create Word files in the 1025Manuscript layout."""

import os
import re
import shutil
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
REFERENCE = ROOT / "templates" / "1025Manuscript_ICH_reference.docx"

TABLE_SPECS = {
    1: (
        "table1_cohort.md",
        "Baseline characteristics by three-class outcome among stays with a mapped outcome.",
        (1.0, 1.25, 0.7, 1.1, 0.7, 1.75),
    ),
    2: (
        "table2_landmarks.md",
        "Landmark populations and outcome distributions at 24, 48, and 72 h.",
        (1.15, 0.8, 0.85, 0.95, 1.0, 1.0),
    ),
    3: (
        "table3_discrimination.md",
        "Landmark discrimination by model and validation design.",
        (1.55, 1.55, 1.1, 1.1, 1.1),
    ),
    4: (
        "table4_stability.md",
        "Prediction stability and 24 h confidence strata. "
        "Differences were calculated from unrounded estimates.",
        (0.6, 0.9, 0.85, 0.75, 0.75, 0.65, 0.7, 0.65, 0.65),
    ),
    5: (
        "table5_sensitivity.md",
        "Prespecified and source-recovered sensitivity analyses.",
        (0.75, 2.0, 0.9, 0.9, 0.95, 0.95),
    ),
    6: (
        "table6_transport.md",
        "Six-direction fixed-weight transport across the three databases at 72 h.",
        (1.0, 1.15, 0.55, 0.6, 0.58, 0.52, 0.5, 0.55, 0.55, 0.5),
    ),
    7: (
        "table7_recalibration_reassessment.md",
        "Direction-specific local recalibration and retrospective "
        "reassessment-trigger simulation with enhanced monitoring through 72 h.",
        (1.2, 0.8, 0.55, 0.6, 0.45, 0.6, 0.5, 0.55, 0.55, 0.55),
    ),
    8: (
        "table8_shared_fold_metrics.md",
        "Complete-cohort shared-fold XGBoost metrics by landmark.",
        (0.85, 0.55, 0.5, 0.65, 0.65, 0.65, 0.6, 0.65, 0.8),
    ),
    9: (
        "table9_hospital_validation.md",
        "Same-row comparison of patient-grouped and hospital-held-out "
        "validation in eICU-CRD.",
        (1.45, 1.05, 1.10, 1.75, 1.15),
    ),
    10: (
        "table10_clinical_transitions.md",
        "Predicted transitions and movement relative to the observed final class "
        "for "
        "shared-fold XGBoost and DG-GRU variants.",
        (0.75, 1.15, 0.50, 0.65, 0.65, 0.70, 0.72, 0.72, 0.61),
    ),
}
TABLE_FONT_SIZES = {4: 8.5, 5: 8.5, 6: 7.8, 7: 8.0, 8: 8.0, 9: 8.5, 10: 8.0}
TABLE_DISPLAY_TO_SOURCE = {
    1: 1,
    2: 2,
    3: 4,
    4: 9,
    5: 10,
    6: 3,
    7: 8,
    8: 6,
    9: 7,
    10: 5,
}
FIGURE_DISPLAY_TO_SOURCE = {
    1: 1,
    2: 2,
    3: 5,
    4: 11,
    5: 3,
    6: 6,
    7: 4,
    8: 10,
    9: 7,
    10: 8,
    11: 9,
}
AUTHOR_LINE = (
    "Zhipeng Wang1*, Luning Wang2*, Changsong Wang1, Xiyu Xu3, "
    "Pengli Zhai4, Zejun Liu1, Hui Feng1, Qian Hou1, Ming Guo1"
)
AUTHOR_METADATA = (
    "Zhipeng Wang; Luning Wang; Changsong Wang; Xiyu Xu; Pengli Zhai; "
    "Zejun Liu; Hui Feng; Qian Hou; Ming Guo"
)


def clear_document_body(document: Document) -> None:
    body = document.element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)
    for section in document.sections:
        for part in (section.header, section.footer):
            for paragraph in part.paragraphs:
                paragraph.clear()
            for table in part.tables:
                table._element.getparent().remove(table._element)


def set_run_font(run, size: float = 12, bold: bool | None = None) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def configure_styles(document: Document) -> None:
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(12)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.line_spacing = 1.5

    for name, size in (("SectionTitle", 14), ("SubsectionTitle", 12)):
        try:
            style = document.styles[name]
        except KeyError:
            style = document.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        style.base_style = normal
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(0)
        style.paragraph_format.space_after = Pt(0)
        style.paragraph_format.line_spacing = 1.5
        style.paragraph_format.keep_with_next = True
        paragraph_properties = style.element.get_or_add_pPr()
        for outline_level in paragraph_properties.findall(qn("w:outlineLvl")):
            paragraph_properties.remove(outline_level)

    try:
        reference = document.styles["RefEntry"]
    except KeyError:
        reference = document.styles.add_style("RefEntry", WD_STYLE_TYPE.PARAGRAPH)
    reference.base_style = normal
    reference.font.name = "Times New Roman"
    reference._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    reference.font.size = Pt(12)
    reference.paragraph_format.left_indent = Inches(0)
    reference.paragraph_format.first_line_indent = Inches(-0.2)
    reference.paragraph_format.line_spacing = 1.5
    reference.paragraph_format.space_after = Pt(0)
    reference.paragraph_format.space_before = Pt(0)


def configure_page(document: Document) -> None:
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.header_distance = Inches(0.5)
    section.footer_distance = Inches(0.5)
    for part in (section.header, section.footer):
        for paragraph in part.paragraphs:
            paragraph.clear()


def add_title(document: Document, title: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(title)
    set_run_font(run, 18, True)


def add_reference_bookmark(paragraph, number: int, run) -> None:
    bookmark_id = str(number)
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), bookmark_id)
    start.set(qn("w:name"), f"_Ref{number}")
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), bookmark_id)
    run._r.addprevious(start)
    run._r.addnext(end)


def add_reference_field(paragraph, number: int) -> None:
    begin_run = paragraph.add_run()
    set_run_font(begin_run, 12)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    begin_run._r.append(begin)

    instruction_run = paragraph.add_run()
    set_run_font(instruction_run, 12)
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = f" REF _Ref{number} \\h "
    instruction_run._r.append(instruction)

    separate_run = paragraph.add_run()
    set_run_font(separate_run, 12)
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    separate_run._r.append(separate)

    result_run = paragraph.add_run(str(number))
    set_run_font(result_run, 12)

    end_run = paragraph.add_run()
    set_run_font(end_run, 12)
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    end_run._r.append(end)


def add_citation_fields(paragraph, token: str) -> None:
    opening = paragraph.add_run("[")
    set_run_font(opening, 12)
    groups = token.strip("[]").split(",")
    for group_index, group in enumerate(groups):
        if group_index:
            comma = paragraph.add_run(",")
            set_run_font(comma, 12)
        if "-" in group:
            start, end = (int(value) for value in group.split("-", 1))
            add_reference_field(paragraph, start)
            hyphen = paragraph.add_run("-")
            set_run_font(hyphen, 12)
            add_reference_field(paragraph, end)
        else:
            add_reference_field(paragraph, int(group))
    closing = paragraph.add_run("]")
    set_run_font(closing, 12)


def add_inline_markup(paragraph, text: str) -> None:
    pattern = re.compile(
        r"(\[[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*\]"
        r"|\*\*.*?\*\*|(?<!\\)\*[^*\n]*?(?<!\\)\*|`.*?`)"
    )
    for token in pattern.split(text):
        if not token:
            continue
        if re.fullmatch(
            r"\[[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*\]",
            token,
        ):
            add_citation_fields(paragraph, token)
            continue
        literal = token.replace(r"\*", "*")
        if token.startswith("**") and token.endswith("**"):
            run = paragraph.add_run(literal[2:-2])
            set_run_font(run, 12, True)
        elif token.startswith("*") and token.endswith("*"):
            run = paragraph.add_run(literal[1:-1])
            set_run_font(run, 12)
            run.italic = True
        elif token.startswith("`") and token.endswith("`"):
            run = paragraph.add_run(literal[1:-1])
            set_run_font(run, 12)
        else:
            run = paragraph.add_run(literal)
            set_run_font(run, 12)


def add_body_paragraph(document: Document, text: str, style: str | None = None) -> None:
    paragraph = document.add_paragraph(style=style or "Normal")
    paragraph.alignment = None
    paragraph.paragraph_format.first_line_indent = Inches(0)
    paragraph.paragraph_format.line_spacing = 1.5
    add_inline_markup(paragraph, text)
    paragraph.paragraph_format.keep_together = False


def add_reference(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="RefEntry")
    match = re.match(r"^(\d+)(\.\s*)(.*)$", text)
    if not match:
        add_inline_markup(paragraph, text)
        return
    number = int(match.group(1))
    number_run = paragraph.add_run(match.group(1))
    set_run_font(number_run, 12)
    add_reference_bookmark(paragraph, number, number_run)
    remainder = paragraph.add_run(match.group(2) + match.group(3))
    set_run_font(remainder, 12)


def add_heading(document: Document, text: str, level: int) -> None:
    style_name = "SectionTitle" if level == 1 else "SubsectionTitle"
    paragraph = document.add_paragraph(style=style_name)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.5
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    set_run_font(run, 14 if level == 1 else 12, True)


def insert_pending_assets(
    document: Document,
    text: str,
    inserted_tables: set[int],
    inserted_figures: set[int],
) -> None:
    table_numbers = [
        int(number)
        for number in re.findall(
            r"(?<!Supplementary )Table\s+([1-9][0-9]*)\b", text
        )
    ]
    if table_numbers:
        for number in sorted(set(table_numbers)):
            if number not in inserted_tables:
                source_number = TABLE_DISPLAY_TO_SOURCE[number]
                filename, caption, widths = TABLE_SPECS[source_number]
                add_table(
                    document,
                    (TABLES / filename).read_text(encoding="utf-8"),
                    f"Table {number}.",
                    caption,
                    widths,
                    font_size=TABLE_FONT_SIZES.get(source_number),
                )
                inserted_tables.add(number)

    figure_numbers = [
        int(number)
        for number in re.findall(r"Figure\s+([1-9][0-9]*)(?![0-9])", text)
    ]
    for number in sorted(set(figure_numbers)):
        if number not in inserted_figures:
            add_figure(document, number)
            inserted_figures.add(number)


def add_markdown(document: Document, markdown: str) -> None:
    lines = markdown.splitlines()
    index = 0
    in_references = False
    inserted_tables: set[int] = set()
    inserted_figures: set[int] = set()
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            add_title(document, line[2:].strip())
            index += 1
            continue
        if line.startswith("### "):
            heading = line[4:].strip()
            add_heading(document, heading, 2)
            index += 1
            continue
        if line.startswith("## "):
            heading = line[3:].strip()
            add_heading(document, heading, 1)
            in_references = (
                heading.lower() == "references"
                or heading.lower().endswith(" references")
            )
            index += 1
            continue
        if re.fullmatch(r"\*\*[^*].*?\*\*", line):
            add_body_paragraph(document, line)
            index += 1
            continue
        if line.startswith("**Background:**") or line.startswith(
            ("**Methods:**", "**Results:**", "**Conclusions:**", "**Keywords:**")
        ):
            paragraph_lines = [line]
            index += 1
            while index < len(lines):
                candidate = lines[index].rstrip()
                if not candidate or candidate.startswith("#"):
                    break
                paragraph_lines.append(candidate)
                index += 1
            paragraph = document.add_paragraph(style="Normal")
            paragraph.alignment = None
            paragraph.paragraph_format.line_spacing = 1.5
            add_inline_markup(paragraph, " ".join(paragraph_lines))
            continue
        paragraph_lines = [line]
        index += 1
        while index < len(lines):
            candidate = lines[index].rstrip()
            if (
                not candidate
                or candidate.startswith("#")
                or candidate.startswith("**")
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        text = " ".join(paragraph_lines)
        if in_references:
            add_reference(document, text)
        else:
            add_body_paragraph(document, text)
            insert_pending_assets(
                document, text, inserted_tables, inserted_figures
            )


def parse_markdown_table(text: str) -> tuple[list[str], list[list[str]]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    rows = []
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        raise ValueError("Empty markdown table")
    return rows[0], rows[1:]


def repeat_header(row) -> None:
    properties = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    properties.insert_element_before(
        header,
        "w:tblCellSpacing",
        "w:jc",
        "w:hidden",
        "w:ins",
        "w:del",
        "w:trPrChange",
    )


def set_table_borders(table) -> None:
    properties = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, value in (
        ("top", "single"),
        ("left", "none"),
        ("bottom", "single"),
        ("right", "none"),
        ("insideH", "none"),
        ("insideV", "none"),
    ):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), value)
        element.set(qn("w:sz"), "12" if value == "single" else "0")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "000000")
        borders.append(element)
    properties.insert_element_before(
        borders,
        "w:shd",
        "w:tblLayout",
        "w:tblCellMar",
        "w:tblLook",
        "w:tblCaption",
        "w:tblDescription",
        "w:tblPrChange",
    )


def set_row_bottom_border(row) -> None:
    for cell in row.cells:
        properties = cell._tc.get_or_add_tcPr()
        borders = OxmlElement("w:tcBorders")
        element = OxmlElement("w:bottom")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "12")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "000000")
        borders.append(element)
        properties.insert_element_before(
            borders,
            "w:shd",
            "w:noWrap",
            "w:tcMar",
            "w:textDirection",
            "w:tcFitText",
            "w:vAlign",
            "w:hideMark",
            "w:headers",
            "w:cellIns",
            "w:cellDel",
            "w:cellMerge",
            "w:tcPrChange",
        )


def normalize_settings(document: Document) -> None:
    """Supply attributes required by the OOXML settings schema."""
    for zoom in document.settings.element.findall(qn("w:zoom")):
        zoom.set(qn("w:percent"), "100")


def add_caption(
    document: Document,
    label: str,
    caption: str,
    *,
    keep_with_next: bool,
) -> None:
    paragraph = document.add_paragraph(style="Normal")
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.5
    paragraph.paragraph_format.keep_with_next = keep_with_next
    paragraph.paragraph_format.keep_together = True
    label_run = paragraph.add_run(label)
    set_run_font(label_run, 12, True)
    caption_run = paragraph.add_run(caption)
    set_run_font(caption_run, 12)


def normalize_table_value(value: str) -> str:
    """Normalize P-value formatting without re-rounding numeric table cells."""
    protected: dict[str, str] = {}

    def protect(text: str) -> str:
        token = f"\u0000{len(protected)}\u0000"
        protected[token] = text
        return token

    def normalize_p(match: re.Match) -> str:
        operator = match.group(1)
        number = Decimal(match.group(2))
        if number < Decimal("0.001") and operator in {"<", "=", "≤"}:
            return protect("P<0.001")
        rounded = number.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        return protect(f"P{operator}{rounded:.3f}")

    value = re.sub(
        r"\b[Pp]\s*(<|≤|=|>|≥)\s*(\d*\.\d+)",
        normalize_p,
        value,
    )

    for token, text in protected.items():
        value = value.replace(token, text)
    return value


def set_cell_text(cell, text: str, *, bold: bool, size: float) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.line_spacing = 1.5
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(2)
    run = paragraph.add_run(normalize_table_value(text))
    set_run_font(run, size=size, bold=bold)


def add_table(
    document: Document,
    markdown: str,
    label: str,
    caption: str,
    widths: tuple[float, ...] | None = None,
    font_size: float | None = None,
) -> None:
    add_caption(document, f"{label} ", caption, keep_with_next=True)

    headers, rows = parse_markdown_table(markdown)
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.allow_autofit = False
    set_table_borders(table)
    if widths is None:
        widths = tuple([6.5 / len(headers)] * len(headers))
    for index, width in enumerate(widths):
        table.columns[index].width = Inches(width)
        for row in table.rows:
            row.cells[index].width = Inches(width)
    body_size = font_size or (
        9.0 if len(headers) <= 6 else (8.0 if len(headers) == 7 else 7.2)
    )
    header_cells = table.rows[0].cells
    for index, value in enumerate(headers):
        header_cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_text(header_cells[index], value, bold=True, size=body_size)
        header_cells[index].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    repeat_header(table.rows[0])
    set_row_bottom_border(table.rows[0])
    for values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_text(cells[index], value, bold=False, size=body_size)
            cells[index].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.keep_with_next = True
    spacer = document.add_paragraph(style="Normal")
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.space_after = Pt(0)
    spacer.paragraph_format.line_spacing = 1.0


FIGURE_CAPTIONS = {
    1: (
        "figure1_cohort_outcome.png",
        "Study population and outcome distribution. "
        "Panel A. Selection of complete-window ICU stays in MIMIC-IV, eICU-CRD, "
        "and MIMIC-III. "
        "Panel B. Distribution of home, facility, and in-hospital death outcomes in the "
        "72 h complete-window cohorts.",
    ),
    2: (
        "figure2_model.png",
        "Dynamic graph gated recurrent unit. The model combines an "
        "eight-channel hourly sequence, a learned feature graph, a 64-unit GRU, "
        "shared ordinal thresholds, and temporal smoothness/non-worsening penalties. "
        "The displayed home < facility < death threshold order is a modeling code, "
        "not a validated severity scale.",
    ),
    3: (
        "figure3_discrimination.png",
        "Landmark discrimination. Panel A. XGBoost series for MIMIC-IV and eICU-CRD "
        "with patient-grouped folds, eICU-CRD with hospital-held-out folds, and "
        "MIMIC-III same-institution cross-era replication. Panel B. Regularized "
        "ordinal DG-GRU for MIMIC-IV and eICU-CRD; error bars show seed-level SD.",
    ),
    4: (
        "figure4_calibration.png",
        "Calibration at 72 h. Reliability curves for home, facility, and "
        "in-hospital death compare XGBoost with the regularized ordinal DG-GRU in "
        "MIMIC-IV and eICU-CRD; MIMIC-III shows XGBoost only.",
    ),
    5: (
        "figure5_stability.png",
        "Prediction stability. Panel A. 24-to-72 h class flip for shared-fold XGBoost "
        "and DG-GRU; XGBoost intervals are 95% evaluation-sample bootstrap intervals "
        "conditional on the fitted models and shared folds, and DG-GRU intervals "
        "show seed-level SD. "
        "Panel B. Flip within XGBoost strata defined by confidence "
        "available at 24 h. "
        "Panel C. Class flip versus probability total variation, with XGBoost shown "
        "as circles and regularized ordinal DG-GRU as squares. Panel D. Row-normalized XGBoost "
        "24-to-72 h transition matrices, shown separately by database. Panel E. Complete "
        "24, 48, and 72 h prediction trajectories classified as stable at all three "
        "landmarks, transient 48 h excursion, successive higher-coded, successive lower-coded, "
        "or other non-successive patterns. Higher- and lower-coded directions reflect "
        "the analysis coding order, not a validated clinical severity gradient.",
    ),
    6: (
        "figure6_ablation.png",
        "Paired DG-GRU contrasts. Panel A. Differences in 72 h macro-AUC. "
        "Panel B. Differences in the 24-to-72 h class-flip rate in percentage "
        "points. Both panels "
        "compare temporal-penalty and ordinal-readout configurations. Error bars "
        "show SD across three seeds; they are not confidence intervals.",
    ),
    7: (
        "figure7_transport_reassessment.png",
        "Six-direction fixed-weight transport and local recalibration. Panel A. "
        "Fixed-weight 72 h macro-AUC and top-label expected calibration error "
        "across all six directional comparisons among MIMIC-IV, MIMIC-III, and "
        "eICU-CRD. Panel B. Multinomial logistic recalibration learning "
        "curves by transport direction. Panel C. One-parameter temperature-scaling "
        "learning curves on the same splits and labels. Curves show 0, 25, 50, 100, "
        "and 200 labelled target outcomes, with one target stay per patient and "
        "disjoint calibration and evaluation patients. Panel D. Paired temperature-"
        "minus-multinomial differences in 72 h expected calibration error at 25 labels.",
    ),
    8: (
        "figure8_reassessment_simulation.png",
        "Retrospective reassessment-trigger simulation. Panel A. Inclusion of predicted "
        "non-death-to-death class conversions versus monitoring retained when enhanced "
        "monitoring through 72 h is selected by the predicted risk of a 24-to-72 h "
        "endpoint class change. Panel B. The corresponding trade-off for a trigger trained "
        "directly for predicted non-death-to-death class conversion. Thresholds are 0.05, "
        "0.10, 0.15, 0.20, and 0.30; open markers indicate 0.10. Panel C. Inclusion of "
        "endpoint class changes and predicted conversions at threshold 0.10, comparing the "
        "two trigger models. Panel D. Fewer stays selected for enhanced monitoring and the "
        "fraction of selected stays without a predicted conversion at threshold 0.10. "
        "Panel E. Out-of-fold "
        "trigger discrimination for endpoint class change and predicted conversion, with 95% "
        "cluster-bootstrap intervals. "
        "Panel F. Corresponding area under the precision-recall curve. Predictions are "
        "out-of-fold within each database. This is a retrospective decision simulation "
        "and does not estimate effects on clinical workflow, placement, length of stay, "
        "or patient outcomes.",
    ),
    9: (
        "figure9_sensitivity_analyses.png",
        "Sensitivity analyses and model-structure stress tests. Panel A. XGBoost macro-AUC "
        "at 24 and 48 h in the full landmark and complete-window populations; open circles "
        "show full landmarks and filled circles show complete windows. Panel B. XGBoost "
        "72 h macro-AUC with three versus four outcome classes. Panel C. Six-to-48 h "
        "macro-AUC on the complete subset; labels show the 6-to-48 h class-flip rate. "
        "Panel D. XGBoost 24-to-72 h class flip with 95% evaluation-sample bootstrap "
        "intervals for the primary cohort, ICU-death exclusion, and MIMIC-IV ICD-era "
        "strata. Panel E. Three-seed coding-order pressure test for the ordinal model; open "
        "circles are individual seeds and diamonds show mean +/- SD for paired "
        "reversed-order minus standard-order differences in AUC in the top panel and "
        "class flip percentage points in the bottom panel. This is not a second clinical outcome ordering. "
        "Panel F. Exploratory continuous NRI and IDI for the ordinal versus nominal head "
        "without temporal regularization, with 95% evaluation-sample bootstrap intervals "
        "and no multiplicity adjustment. In eICU-CRD, evaluation-sample intervals "
        "resample hospitals.",
    ),
    10: (
        "figure10_hospital_validation.png",
        "eICU-CRD hospital-held-out validation. Panel A. Patient-grouped and "
        "hospital-held-out macro-AUC on the same stays at each landmark. "
        "Panel B. Validation cohort construction, including exclusion of patients "
        "observed in more than one hospital. Panel C. Hospital-held-out minus "
        "patient-grouped differences in percentage points for class flips, "
        "predicted non-death-to-death class conversions, "
        "and home-to-facility transitions, with 95% cluster-bootstrap intervals. "
        "Panel D. Distribution of 72 h macro-AUC across evaluable hospitals, with "
        "median, P10-P90, and DerSimonian-Laird random-effects summary. This is "
        "retrospective internal-external validation, not prospective validation.",
    ),
    11: (
        "figure11_clinical_transitions_safety.png",
        "Predicted transitions and temporal-penalty transition decomposition. "
        "Panel A. Predicted non-death-to-death class conversions and home-to-facility transitions across "
        "the three databases for shared-fold XGBoost. Panel B. Movement toward, away from, "
        "or equally distant from the observed final class among stays whose predicted class changed. "
        "Panel C. Percentage "
        "of changed predictions with an increased probability for the eventual "
        "class; labels show the mean change in that probability. Panel D. Temporal "
        "minus no-temporal changes in movement toward or away from the observed final class "
        "for higher- or lower-coded changes and in predicted conversions or missed deaths, "
        "with 95% paired "
        "bootstrap intervals. The temporal penalty has no clinical validation.",
    ),
}


def add_figure(document: Document, number: int) -> None:
    source_number = FIGURE_DISPLAY_TO_SOURCE[number]
    filename, caption = FIGURE_CAPTIONS[source_number]
    path = FIGURES / filename
    if not path.exists():
        raise FileNotFoundError(path)
    paragraph = document.add_paragraph(style="Normal")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.5
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.keep_together = True
    paragraph.add_run().add_picture(str(path), width=Inches(6.4))
    add_caption(document, f"Figure {number}. ", caption, keep_with_next=False)


def build_main_document() -> Path:
    title = (
        "Predicted-Class Stability and Transport of Discharge-Disposition Models "
        "Among ICU Stays With Sepsis at 24, 48, and 72 Hours: A Three-Database "
        "Study of Recalibration and Enhanced-Monitoring Trigger Simulation"
    )
    document = Document()
    configure_page(document)
    configure_styles(document)
    document.core_properties.title = title
    document.core_properties.subject = "Sepsis discharge-destination prediction stability"
    document.core_properties.author = AUTHOR_METADATA
    document.core_properties.last_modified_by = "Zhipeng Wang"
    document.core_properties.keywords = (
        "sepsis; discharge destination; ordinal outcome; prediction stability; "
        "temporal regularization"
    )

    markdown = (OUTPUTS / "manuscript_sepsis_stability.md").read_text(encoding="utf-8")
    if AUTHOR_LINE.replace("*", r"\*") not in markdown:
        raise ValueError("Assembled manuscript is missing the synchronized author block")
    add_markdown(document, markdown)

    output = OUTPUTS / "1029TP_Sepsis_Manuscript.docx"
    normalize_settings(document)
    document.save(output)
    return output


def build_supplement() -> Path:
    title = (
        "Predicted-Class Stability and Transport of Discharge-Disposition Models "
        "Among ICU Stays With Sepsis at 24, 48, and 72 Hours: A Three-Database "
        "Study of Recalibration and Enhanced-Monitoring Trigger Simulation"
    )
    document = Document()
    configure_page(document)
    configure_styles(document)
    document.core_properties.title = f"{title} - Supplementary Material"
    document.core_properties.author = AUTHOR_METADATA
    document.core_properties.last_modified_by = "Zhipeng Wang"
    add_title(document, title)
    add_heading(document, "Supplementary Material", 1)
    add_body_paragraph(
        document,
        "This supplement reports decision-curve coordinates, seed-level paired "
        "DG-GRU contrasts, additional model metrics, calibration summaries, and "
        "continuous ordinal reclassification metrics. It also reports major "
        "predicted-class transitions and movement relative to the observed final class, "
        "the temporal-penalty transition decomposition, and "
        "fold-level shared-fold XGBoost metrics, hospital-held-out validation, "
        "full six-direction transport, direction-specific local recalibration, "
        "retrospective reassessment-trigger simulations, and an item-level "
        "TRIPOD+AI reporting checklist.",
    )
    supplement_specs = (
        (
            "Supplementary Table S1.",
            "supplementary_table_s6_clinical_flips.md",
            "Major predicted-class transition direction and temporal-penalty "
            "decomposition. For temporal-minus-no-temporal rows, the category-change "
            "column is closer-higher / farther-higher / closer-lower / farther-lower. "
            "The final column reports the change in new predicted non-death-to-death "
            "conversions and deaths not predicted at 72 h.",
        ),
        (
            "Supplementary Table S2.",
            "supplementary_table_s7_shared_fold.md",
            "Fold-level shared-fold XGBoost metrics by landmark.",
        ),
        (
            "Supplementary Table S3.",
            "supplementary_table_s4_calibration.md",
            "One-versus-rest calibration slope and intercept "
            "at 72 h.",
        ),
        (
            "Supplementary Table S4.",
            "supplementary_table_s1_decision_curves.md",
            "Decision-curve coordinates at 72 h for home "
            "discharge and in-hospital death.",
        ),
        (
            "Supplementary Table S5.",
            "supplementary_table_s8_hospital_validation.md",
            "eICU-CRD hospital-held-out validation, including hospital-level "
            "discrimination, calibration, stability, and random-effects summary.",
        ),
        (
            "Supplementary Table S6.",
            "supplementary_table_s2_contrasts.md",
            "Paired DG-GRU contrasts across three seeds.",
        ),
        (
            "Supplementary Table S7.",
            "supplementary_table_s3_model_metrics.md",
            "Additional 72 h model metrics.",
        ),
        (
            "Supplementary Table S8.",
            "supplementary_table_s5_ordinal_clinical.md",
            "Continuous NRI and IDI for ordinal versus "
            "nominal readouts at 72 h.",
        ),
        (
            "Supplementary Table S9.",
            "supplementary_table_s9_transport_full.md",
            "Full landmark results for all six fixed-weight transport directions "
            "across MIMIC-IV, MIMIC-III, and eICU-CRD.",
        ),
        (
            "Supplementary Table S10.",
            "supplementary_table_s10_recalibration.md",
            "Direction-specific local recalibration learning curves with "
            "0, 25, 50, 100, and 200 target labels, using one stay per patient "
            "and disjoint calibration and evaluation patients. Ranges are 2.5th "
            "and 97.5th percentiles across repeated calibration/evaluation splits.",
        ),
        (
            "Supplementary Table S11.",
            "supplementary_table_s11_reassessment.md",
            "Retrospective reassessment-trigger simulation for endpoint class change "
            "and direct conversion triggers across thresholds and databases.",
        ),
        (
            "Supplementary Table S12.",
            "supplementary_table_s12_tripod_ai.md",
            "TRIPOD+AI checklist with item-level section mapping and disclosed gaps.",
        ),
    )
    for label, filename, caption in supplement_specs:
        path = TABLES / filename
        if not path.exists():
            raise FileNotFoundError(path)
        add_table(document, path.read_text(encoding="utf-8"), label, caption)
    output = OUTPUTS / "1029TP_Sepsis_Supplementary.docx"
    normalize_settings(document)
    document.save(output)
    return output


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    main_document = build_main_document()
    supplement = build_supplement()
    shutil.copy2(main_document, ROOT / main_document.name)
    shutil.copy2(main_document, ROOT / "1029Manuscript.docx")
    shutil.copy2(supplement, ROOT / supplement.name)
    print(main_document, flush=True)
    print(ROOT / "1029Manuscript.docx", flush=True)
    print(supplement, flush=True)


if __name__ == "__main__":
    main()
