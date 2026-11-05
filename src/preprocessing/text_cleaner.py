from html import unescape
import re
import unicodedata

from lxml import html


ADVERTISEMENT_LINE = re.compile(
    r"^\s*(?:\[광고\]|광고\s*[:：]|ADVERTISEMENT\b|무단\s*전재|무단\s*복제|Copyright\b|ⓒ|저작권자\b).*$",
    re.IGNORECASE,
)


# HTML·광고 전용 행·장식 문자를 제거하되 문단·금융 수치·부정 표현은 보존
def clean_news_text(value: str | None) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    text = unescape(value)
    document = html.fragment_fromstring(text, create_parent="div", parser=html.HTMLParser(no_network=True))
    for element in document.xpath(".//script | .//style | .//noscript"):
        element.drop_tree()
    for element in document.iter():
        if element.tag in ("p", "div", "br", "li", "h1", "h2", "h3"):
            element.tail = "\n" + (element.tail or "")
    text = unicodedata.normalize("NFC", document.text_content())
    text = "".join(character for character in text
                   if character in "\n\t" or unicodedata.category(character) not in ("Cc", "Cf"))
    lines = []
    for line in text.splitlines():
        if ADVERTISEMENT_LINE.match(line):
            continue
        line = re.sub(r"[■□◆◇▶▷●○★☆]", " ", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)
