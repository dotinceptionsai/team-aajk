import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup, Tag


@dataclass
class QA:
    question: str
    answer: str


class EuropcarScraper:
    def __init__(self, url="https://faq.europcar.com/", max_depth=3):
        self.qa = []
        EuropcarScraper._feed_qa_items(self.qa, url, max_depth)

    @staticmethod
    def _feed_qa_items(qas: list[QA], url: str, remaining_depth: int):
        print(f"Visiting {url} ...")
        if remaining_depth == 0:
            return

        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")

        for child_uri in EuropcarScraper._find_qa_page_links(soup):
            time.sleep(1)  # Pause avoiding to be rejected by website
            EuropcarScraper._feed_qa_items(qas, child_uri, remaining_depth - 1)

        qas.extend(EuropcarScraper._extract_qa(soup))

    @staticmethod
    def _find_qa_page_links(soup: BeautifulSoup) -> Iterable[str]:
        return list(EuropcarScraper._faq_big_icon_link(soup)) + list(
            EuropcarScraper._faq_big_simple_link(soup)
        )

    @staticmethod
    def _extract_qa(soup: BeautifulSoup) -> Iterable[QA]:
        answer_divs = soup.find_all("div", class_="dydu_answer")
        for ans_div in answer_divs:
            if h2 := ans_div.find("h2"):
                question = h2.get_text(".", strip=True)
                if div_ans := ans_div.find("div", itemprop="acceptedAnswer"):
                    answer = "\n".join(
                        [p.get_text("\n", strip=True) for p in div_ans.find_all("p")]
                    )
                    yield QA(question, answer)

    @staticmethod
    def _faq_big_icon_link(soup: BeautifulSoup) -> Iterable[str]:
        for div in soup.find_all("div", {"class": "dydu_thematic-icon"}):
            if par := div.parent:
                if par.name == "a":
                    yield par["href"]

    @staticmethod
    def _faq_big_simple_link(soup: BeautifulSoup) -> Iterable[str]:
        for li in soup.find_all("li", {"class": "dydu_knowledge"}):
            if a := li.a:
                yield a["href"]


class FedoraScraper:
    def __init__(self, url="https://fedoraproject.org/wiki/FAQ#Getting_Started"):
        self.qa: list[QA] = []
        FedoraScraper._feed_qa_items(self.qa, url)

    @staticmethod
    def _feed_qa_items(qas: list[QA], url: str):
        print(f"Visiting {url} ...")

        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")

        for q_item in FedoraScraper._find_qa_sections(soup):
            qas.append(FedoraScraper._extract_qa(q_item))

    @staticmethod
    def _find_qa_sections(soup: BeautifulSoup) -> Iterable[BeautifulSoup]:
        for item in soup.find_all("h3"):
            if item.get_text(strip=True).strip().endswith("?"):
                yield item

    @staticmethod
    def _extract_qa(item: Tag) -> QA:
        q = item.get_text(strip=True).strip()
        full_answer = FedoraScraper._extract_answer(item)
        return QA(q, full_answer)

    @staticmethod
    def _extract_answer(tag: Tag) -> str:
        content = []
        next_p = tag
        while (next_p := next_p.find_next_sibling()) and FedoraScraper._is_paragraph(
            next_p
        ):
            answer = next_p.get_text().strip()
            if answer:
                content.append(answer)
        full_answer = "\n".join(content)
        return full_answer

    @staticmethod
    def _is_paragraph(tag: Tag) -> bool:
        return tag is not None and tag.name == "p"


import yaml


def save_scrape(scraper, dest_filename):
    kb_file = Path("/Users/jlinho/Desktop/gr") / dest_filename
    with open(kb_file, "w") as f:
        items = []
        for i in scraper.qa:
            items.append(i.question)
            items.append(i.answer)
        yaml.dump(items, f)


if __name__ == "__main__":
    scrapers = {"europcar": EuropcarScraper, "fedora": FedoraScraper}

    for name, scraper in scrapers.items():
        save_scrape(scraper(), f"{name}.yaml")
