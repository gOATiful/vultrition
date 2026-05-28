from typing import Any

import svgwrite

from vultrition.models.results import AnalysisResults, CompletenessResults, CrossContaminationResults, SplitNumericalMetricsResults, SplitStatisticalMetricsResults, TimeSpanResults



FONT = "Inter, Arial, Helvetica, sans-serif"
W = 1055
H = 1330
MARGIN = 26
BLUE = "#1b4fc7"
DARK = "#0a1733"
TEXT = "#222631"
MUTED = "#586277"
BORDER = "#b9c1cf"
LIGHT_FILL = "#eef4ff"
LIGHTER = "#f6f9ff"
WHITE = "#ffffff"
BLACK = "#000000"

SECTION_OFFSET = 45
QUALITY_METRICS_PADDING = 20
STRUCTURAL_METRICS_PADDING = 24


def format_data(data: Any, percent: bool = False) -> str:
    if isinstance(data, float):
        if percent:
            return f"{data:.2f}%"
        return f"{data:.2f}"
    if isinstance(data, tuple):
        return f"{data[0]} - {data[1]}"
    if data is None:
        "--"
    return str(data)


def headline(text: str, subheadline: str, dwg: svgwrite.Drawing, x: int, y: int):
    dwg.add(dwg.text(text, insert=(x, y), fill=TEXT,
            font_size=54, font_family=FONT, font_weight="bold"))
    if subheadline:
        dwg.add(dwg.text(subheadline, insert=(x+3, y + 25),
                fill=MUTED, font_size=21, font_family=FONT))


def separator(dwg: svgwrite.Drawing, x: int, y: int, width: int, stroke_width: int = 1, stroke_color: str = BORDER):
    dwg.add(dwg.line(start=(x, y), end=(x + width, y),
            stroke=stroke_color, stroke_width=stroke_width))


def data_row(label: str, value: str, dwg: svgwrite.Drawing, x: int, y: int):
    dwg.add(dwg.text(label, insert=(x, y), fill=TEXT,
            font_size=24, font_family=FONT, font_weight="bold"))

    value_text = dwg.text(format_data(value), insert=(W - MARGIN, y), fill=TEXT,
                          # becomes text-anchor="end"
                          font_size=24, font_family=FONT,         text_anchor="end",
                          )
    dwg.add(value_text)


def subheadline(text: str, dwg: svgwrite.Drawing, x: int, y: int):
    dwg.add(dwg.text(text, insert=(x, y), fill=BLACK,
            font_size=32, font_family=FONT, font_weight="bold"))


def splitnumerical_row(label: str, description: str, value: SplitNumericalMetricsResults, dwg: svgwrite.Drawing, x: int, y: int, percent: bool = False):
    dwg.add(dwg.text(label, insert=(x, y+12), fill=TEXT,
            font_size=24, font_family=FONT, font_weight="bold"))
    dwg.add(dwg.text(description, insert=(x, y + 35), fill=MUTED,
            font_size=18, font_family=FONT))

    x = W - (W - MARGIN) * 3 / 5

    section_width = (W - x) / 4
    
    sections = ["Train", "Test", "Validation", "Overall"]
    if isinstance(value, CompletenessResults):
        values = [value.train, value.test, value.valid, value.overall]
    else:
        values = [value.train, value.test, value.validation, value.overall]
        

    for i, (section, val) in enumerate(zip(sections, values)):
        if val == -1 or val is None:
            val = "--"
        section_x = x + i * section_width
        dwg.add(dwg.text(section, insert=(section_x, y), fill=MUTED,
                font_size=18, font_family=FONT, text_anchor="middle", font_weight="bold" if section == "Overall" else "normal"))
        dwg.add(dwg.text(format_data(val, percent=percent), insert=(section_x, y + 25), fill=TEXT,
                font_size=24, font_family=FONT, text_anchor="middle", font_weight="bold" if section == "Overall" else "normal"))
        if i < len(sections) - 1:
            dwg.add(dwg.line(start=(section_x + section_width/2,  y-10), end=(section_x + section_width/2,  y + 35),
                             stroke=BORDER, stroke_width=2))


def timespan_row(label: str, description: str, value: TimeSpanResults, dwg: svgwrite.Drawing, x: int, y: int):
    dwg.add(dwg.text(label, insert=(x, y+12), fill=TEXT,
            font_size=24, font_family=FONT, font_weight="bold"))
    dwg.add(dwg.text(description, insert=(x, y + 35), fill=MUTED,
            font_size=18, font_family=FONT))

    x = W - (W - MARGIN) * 3 / 5

    section_width = (W - x) / 4

    sections = ["Train", "Test", "Validation", "Overall"]
    values = [value.train, value.test, value.validation, value.overall]

    for i, (section, val) in enumerate(zip(sections, values)):
        if val == -1 or val is None:
            val = "--"
        section_x = x + i * section_width
        dwg.add(dwg.text(section, insert=(section_x, y), fill=MUTED,
                font_size=18, font_family=FONT, text_anchor="middle", font_weight="bold" if section == "Overall" else "normal"))
        dwg.add(dwg.text(format_data(val), insert=(section_x, y + 25), fill=TEXT,
                font_size=24, font_family=FONT, text_anchor="middle", font_weight="bold" if section == "Overall" else "normal"))
        if i < len(sections) - 1:
            dwg.add(dwg.line(start=(section_x + section_width/2,  y-10), end=(section_x + section_width/2,  y + 35),
                             stroke=BORDER, stroke_width=2))



def cross_similarity_row(label: str, description: str, value: CrossContaminationResults, dwg: svgwrite.Drawing, x: int, y: int):
    dwg.add(dwg.text(label, insert=(x, y+12), fill=TEXT,
            font_size=24, font_family=FONT, font_weight="bold"))
    dwg.add(dwg.text(description, insert=(x, y + 35), fill=MUTED,
            font_size=18, font_family=FONT))

    x = W - (W - MARGIN) * 3 / 5

    section_width = (W - x) / 4

    sections = [("Train","Test"), ("Train","Validation"), ("Test","Validation")]
    values = [value.train_test, value.train_valid, value.test_valid]

    for i, (section, val) in enumerate(zip(sections, values)):
        section_x = x + (i+1) * section_width
        dwg.add(dwg.text(section[0] + "-" + section[1], insert=(section_x, y), fill=MUTED,
                font_size=15, font_family=FONT, text_anchor="middle"))
        dwg.add(dwg.text(format_data(val, percent=False), insert=(section_x, y + 20), fill=TEXT,
                font_size=20, font_family=FONT, text_anchor="middle"))

        if i < len(sections) - 1:
            dwg.add(dwg.line(start=(section_x + section_width/2,  y-10), end=(section_x + section_width/2,  y + 35),
                             stroke=BORDER, stroke_width=2))



def cross_contamination_row(label: str, description: str, value_a_b: CrossContaminationResults, value_b_a, dwg: svgwrite.Drawing, x: int, y: int):
    dwg.add(dwg.text(label, insert=(x, y+12), fill=TEXT,
            font_size=24, font_family=FONT, font_weight="bold"))
    dwg.add(dwg.text(description, insert=(x, y + 35), fill=MUTED,
            font_size=18, font_family=FONT))

    x = W - (W - MARGIN) * 3 / 5

    section_width = (W - x) / 4

    sections = [("Train","Test"), ("Train","Validation"), ("Test","Validation")]
    values_a_b = [value_a_b.train_test * 100, value_a_b.train_valid * 100, value_a_b.test_valid* 100]
    values_b_a = [value_b_a.train_test*100, value_b_a.train_valid*100, value_b_a.test_valid*100]

    for i, (section, val_a_b, val_b_a) in enumerate(zip(sections, values_a_b, values_b_a)):
        section_x = x + (i+1) * section_width
        dwg.add(dwg.text(section[0] + " -> " + section[1], insert=(section_x, y), fill=MUTED,
                font_size=15, font_family=FONT, text_anchor="middle"))
        dwg.add(dwg.text(section[0] + " -> " + section[1], insert=(section_x, y), fill=MUTED,
                font_size=15, font_family=FONT, text_anchor="middle"))
        dwg.add(dwg.text(format_data(val_a_b, percent=True), insert=(section_x, y + 20), fill=TEXT,
                font_size=20, font_family=FONT, text_anchor="middle"))
        dwg.add(dwg.text(section[0] + " <- " + section[1], insert=(section_x, y + 40), fill=MUTED,
                font_size=15, font_family=FONT, text_anchor="middle"))
        dwg.add(dwg.text(format_data(val_b_a, percent=True), insert=(section_x, y + 60), fill=TEXT,
                font_size=20, font_family=FONT, text_anchor="middle"))
        if i < len(sections) - 1:
            dwg.add(dwg.line(start=(section_x + section_width/2,  y-10), end=(section_x + section_width/2,  y + 35),
                             stroke=BORDER, stroke_width=2))



def data_row_split(label_value_pairs: list[tuple[str, str]], dwg: svgwrite.Drawing, x: int, y: int):
    section_width = (W - MARGIN) / len(label_value_pairs)
    
    for i, (label, value) in enumerate(label_value_pairs):
        dwg.add(dwg.text(label, insert=(x + i * section_width, y), fill=TEXT,
                font_size=24, font_family=FONT, font_weight="bold"))
        dwg.add(dwg.text(format_data(value), insert=(x + i * section_width + section_width - MARGIN, y), fill=TEXT,
                # becomes text-anchor="end"
                font_size=24, font_family=FONT,         text_anchor="end",
                ))
        

def colour_row(label: str, dwg: svgwrite.Drawing, x: int, y: int):
    rect_size = 25
    border = 2
    dwg.add(dwg.rect(insert=(x+border, y),
            size=(W-2*border, rect_size), fill=MUTED))
    dwg.add(dwg.text(label, insert=(x + MARGIN, y+rect_size/2 + 4), fill=WHITE,
            font_size=14, font_family=FONT, font_weight="bold"))


def split_statistical_row(label: str, description: str, value: SplitStatisticalMetricsResults, dwg: svgwrite.Drawing, x: int, y: int):
    dwg.add(dwg.text(label, insert=(x, y+40), fill=TEXT,
            font_size=24, font_family=FONT, font_weight="bold"))
    dwg.add(dwg.text(description, insert=(x, y + 65), fill=MUTED,
            font_size=18, font_family=FONT))

    x = W - (W - MARGIN) * 3 / 5

    section_width = (W - x) / 4

    sections = ["Train", "Test", "Validation", "Overall"]
    values = [value.train, value.test, value.validation, value.overall]

    for i, (section, val) in enumerate(zip(sections, values)):
        y_offset = y

        if val is None or val.min == -1:
            val = "--"
            
        section_x = x + i * section_width

        if i < len(sections) - 1:
            dwg.add(dwg.line(start=(section_x + section_width/2,  y-10), end=(section_x + section_width/2,  y + 80),
                             stroke=BORDER, stroke_width=2))

        dwg.add(dwg.text(section, insert=(section_x, y), fill=MUTED,
                font_size=18, font_family=FONT, text_anchor="middle", font_weight="bold" if section == "Overall" else "normal"))
        if val is None or val == "--":
            dwg.add(dwg.text(val, insert=(section_x, y + 40), fill=BLACK,
                             font_size=18, font_family=FONT))
            continue
        for stat_label, stat_value in [("min:", val.min), ("max:", val.max ), ("mean:", val.mean ), ("std:", val.std )]:
            if stat_value is None or stat_value == -1:
                continue
            y_offset += 18
            dwg.add(dwg.text(stat_label, insert=(section_x - section_width/2 + 7, y_offset), fill=BLACK,
                             font_size=18, font_family=FONT))
            dwg.add(dwg.text(format_data(stat_value), insert=(section_x+section_width/2 - 7, y_offset), fill=BLACK,
                             font_size=18, font_family=FONT, text_anchor="end"))


def add_bottom_line(dwg: svgwrite.Drawing, y: int):
    dwg.add(dwg.text("Generated with Vultrition (https://github.com/gOATiful/vultrition)",
            insert=(MARGIN, y), fill=MUTED, font_size=14, font_family=FONT))


def add_row_separator(dwg: svgwrite.Drawing, y: int):
    y += 15
    separator(dwg, MARGIN, y, W-(2*MARGIN),
              stroke_width=4, stroke_color=BLACK)
    y += 35
    return y


def draw_label(label: AnalysisResults):
    dwg = svgwrite.Drawing(size=(f"{W}px", f"{H}px"), profile='tiny')
    dwg.viewbox(0, 0, W, H)
    bg = dwg.rect(insert=(0, 0), size=(W, H), fill=WHITE)
    bg.stroke(color=BLACK, width=4)
    dwg.add(bg)
    root = dwg.g()
    dwg.add(root)
    headline("Dataset Vultrition Label",
             "Nutrition-Label style summary for software vulnerability datasets", dwg, MARGIN, 60)
    y = 100
    separator(dwg, 0, y, W, stroke_width=20, stroke_color=BLACK)
    y += 40

    data_row("Name:", label.name, dwg, MARGIN, y)
    y = add_row_separator(dwg, y)
    # data_row("Description:", label.description, dwg, MARGIN, y)
    # data_row("License:", label.license, dwg, MARGIN, y)
    data_row_split([("License:", label.license), ("Version:", label.version), ("Languages:", label.languages)], dwg, MARGIN, y)
    y += SECTION_OFFSET
    subheadline("Quality Facts", dwg, MARGIN, y)
    y += 15
    separator(dwg, 0, y, W, stroke_width=10, stroke_color=BLACK)
    y += QUALITY_METRICS_PADDING + 4
    splitnumerical_row(
        "Entries:", "Number of entries in the dataset", label.quality_metrics.samples, dwg, MARGIN, y)
    y += 42
    separator(dwg, MARGIN, y, W-(2*MARGIN),
              stroke_width=4, stroke_color=BLACK)
    y += QUALITY_METRICS_PADDING + 4
    splitnumerical_row(
        "Vulnerability Ratio:", "Ratio between vuln and non-vuln entries", label.quality_metrics.balance, dwg, MARGIN, y)
    y += 42
    separator(dwg, MARGIN, y, W-(2*MARGIN),
              stroke_width=4, stroke_color=BLACK)
    y += QUALITY_METRICS_PADDING + 4
    splitnumerical_row(
        "Completeness:", "Percentage of entries with complete meta data", label.quality_metrics.completeness, dwg, MARGIN, y)
    y += 42
    separator(dwg, MARGIN, y, W-(2*MARGIN),
              stroke_width=4, stroke_color=BLACK)
    y += QUALITY_METRICS_PADDING
    splitnumerical_row(
        "CWEs:", "Number of unique CWE IDs", label.quality_metrics.diversity.unique_cwes, dwg, MARGIN, y)
    y += 42
    separator(dwg, MARGIN, y, W-(2*MARGIN),
              stroke_width=4, stroke_color=BLACK)
    y += QUALITY_METRICS_PADDING
    splitnumerical_row(
        "Projects:", "Number of unique projects", label.quality_metrics.diversity.unique_projects, dwg, MARGIN, y)

    y += 42
    separator(dwg, MARGIN, y, W-(2*MARGIN),
              stroke_width=4, stroke_color=BLACK)
    y += QUALITY_METRICS_PADDING
    timespan_row("Timespan:", "Timespan of data entries in years",
                 label.quality_metrics.timespan, dwg, MARGIN, y)
    y += 42
    separator(dwg, MARGIN, y, W-(2*MARGIN),
              stroke_width=4, stroke_color=BLACK)
    y += QUALITY_METRICS_PADDING
    splitnumerical_row(
        "Similarity (ANS):", "Mean Top-3 cosine similarity", label.quality_metrics.similarity_top3, dwg, MARGIN, y, percent=False)
    y += 42
    separator(dwg, MARGIN, y, W-(2*MARGIN),
              stroke_width=4, stroke_color=BLACK)
    y += QUALITY_METRICS_PADDING
    splitnumerical_row(
        "Near-Duplicates Rate (NDR):", "Percentage of entries with near-duplicates", label.quality_metrics.similar_functions_top3, dwg, MARGIN, y, percent=True)
    y += QUALITY_METRICS_PADDING

    if label.quality_metrics.cross_contamination.train_test != -1:
        y += 42
        separator(dwg, MARGIN, y, W-(2*MARGIN),
                  stroke_width=4, stroke_color=BLACK)
        y += QUALITY_METRICS_PADDING
        cross_similarity_row(
            "Split nearest neighbors similarity :", "Mean average nearest neighbor similarity scores of A and B", label.quality_metrics.cross_contamination, dwg, MARGIN, y)
        y += 42
        separator(dwg, MARGIN, y, W-(2*MARGIN),
                  stroke_width=4, stroke_color=BLACK)
        y += QUALITY_METRICS_PADDING
        cross_contamination_row(
            "Split similarity:", "Percentage of entries in A with near-duplicates in B", label.quality_metrics.cross_contamination_a_b_above_threshold, label.quality_metrics.cross_contamination_b_a_above_threshold, dwg, MARGIN, y)
        
    else:
        pass
    y += SECTION_OFFSET + 35
    subheadline("Structural Facts", dwg, MARGIN, y)
    y += 15
    separator(dwg, 0, y, W, stroke_width=10, stroke_color=BLACK)
    y += STRUCTURAL_METRICS_PADDING
    split_statistical_row(
        "Lines of Code:", "Source code lines per entry", label.structural_metrics.loc, dwg, MARGIN, y)
    y += 90
    separator(dwg, MARGIN, y, W-(2*MARGIN),
              stroke_width=4, stroke_color=BLACK)
    y += STRUCTURAL_METRICS_PADDING
    split_statistical_row(
        "Tokens:", "Tokens per entry", label.structural_metrics.tokens, dwg, MARGIN, y)
    y += 90
    separator(dwg, MARGIN, y, W-(2*MARGIN),
              stroke_width=4, stroke_color=BLACK)
    y += STRUCTURAL_METRICS_PADDING
    split_statistical_row(
        "Cyclomatic Complexity:", "Cyclomatic complexity per entry", label.structural_metrics.cyclomatic_complexity, dwg, MARGIN, y)
    y += 90
    separator(dwg, MARGIN, y, W-(2*MARGIN),
              stroke_width=4, stroke_color=BLACK)
    y += STRUCTURAL_METRICS_PADDING

    add_bottom_line(dwg, y)
    return dwg.tostring()