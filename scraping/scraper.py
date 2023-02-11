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


class NIHScraper:
    def __init__(self, url="https://search.grants.nih.gov/faq/api/faq/664"):
        self.qa: list[QA] = []
        NIHScraper._feed_qa_items(self.qa, url)

    @staticmethod
    def _feed_qa_items(qas: list[QA], url: str):
        print(f"Visiting {url} ...")

        resp = requests.get(url, headers={"Accept": "application/json"})
        body = resp.json()

        for data in body["data"]:
            for header in data["headers"]:
                print(f"Parsing {header['Header_Name']} ...")
                for qa in header["questions"]:
                    question = qa["Question"]
                    a = BeautifulSoup(qa["Answer"].replace("&nbsp;", ""), "html.parser")
                    answer_paragraphs = [p.get_text(strip=False) for p in a.find_all("p")]
                    answer = ""
                    for p in answer_paragraphs:
                        suffix = " " if p.endswith(".") else ". "
                        answer += p + suffix

                    qas.append(QA(question, answer.strip()))


class OlympicsScraper:
    def __init__(self, site="https://olympics.com", root_page="/ioc/faq"):
        self.qa = []

        root_page = requests.get(site + root_page, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(root_page.content, "html.parser")

        pages = [a["href"] for a in soup.find_all("a", {"class": "btn-ioc"}) if
                 a.get_text(strip=True).startswith("View All Questions")]

        for page in pages:
            OlympicsScraper._feed_qa_items(self.qa, site + page)

    @staticmethod
    def _feed_qa_items(qas: list[QA], url: str):
        print(f"Visiting {url} ...")

        page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(page.content, "html.parser")

        for q_item in OlympicsScraper._find_qa_sections(soup):
            qas.append(OlympicsScraper._extract_qa(q_item))

    @staticmethod
    def _find_qa_sections(soup: BeautifulSoup) -> Iterable[BeautifulSoup]:
        for ul in soup.find_all("li"):
            if "data-accordion-item" in ul.attrs:
                yield ul

    @staticmethod
    def _extract_qa(item: Tag) -> QA:
        q_candidate = [d for d in item.find_all("div") if "data-accordion-opener-title" in d.attrs]
        assert len(q_candidate) == 1, "Could not find question div"
        question = q_candidate[0].get_text(strip=True)

        q_answer = [d for d in item.find_all("ul") if "data-accordion-content" in d.attrs]
        assert len(q_answer) == 1, "Could not find answer ul"
        answer = q_answer[0].get_text(strip=True, separator="\n")
        if idx := answer.find("Learn more:"):
            answer = answer[:idx].strip()

        return QA(question, answer)


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


class WwfScraper:
    def __init__(self, url="https://www.wwf.org.uk/faqs"):
        self.qa: list[QA] = []
        WwfScraper._feed_qa_items(self.qa, url)

    @staticmethod
    def _feed_qa_items(qas: list[QA], url: str):
        print(f"Visiting {url} ...")

        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")

        for q_item in WwfScraper._find_qa_sections(soup):
            qas.append(WwfScraper._extract_qa(q_item))

    @staticmethod
    def _find_qa_sections(soup: BeautifulSoup) -> Iterable[BeautifulSoup]:
        for item in soup.find_all("h3", {"class": "faqfield-question"}):
            yield item

    @staticmethod
    def _extract_qa(item: Tag) -> QA:
        q = item.get_text(strip=True).strip()
        full_answer = WwfScraper._extract_answer(item)
        return QA(q, full_answer)

    @staticmethod
    def _extract_answer(tag: Tag) -> str:
        content = []
        answer_div = tag.find_next_sibling()
        assert answer_div.name == "div" and answer_div["class"] == ["faqfield-answer"]

        for p in answer_div.find_all("p"):
            answer = p.get_text().strip()
            if answer:
                content.append(answer + "." if not answer.endswith(".") else answer)

        return "\n".join(content)


class FdaCovidScraper:
    def __init__(self,
                 url="https://www.fda.gov/emergency-preparedness-and-response/coronavirus-disease-2019-covid-19/covid-19-frequently-asked-questions"):
        self.qa: list[QA] = []
        FdaCovidScraper._feed_qa_items(self.qa, url)

    @staticmethod
    def _feed_qa_items(qas: list[QA], url: str):
        print(f"Visiting {url} ...")

        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")

        for q_item in FdaCovidScraper._find_qa_sections(soup):
            qas.append(FdaCovidScraper._extract_qa(q_item))

    @staticmethod
    def _find_qa_sections(soup: BeautifulSoup) -> Iterable[BeautifulSoup]:
        for item in soup.find_all("div", {"class": "fda-accordion-panel"}):
            if item["title"].startswith("Q:"):
                yield item

    @staticmethod
    def _extract_qa(item: Tag) -> QA:
        qdiv, adiv = [child for child in item.children if child.name == "div"]
        full_question = FdaCovidScraper._extract_question(qdiv)
        full_answer = FdaCovidScraper._extract_answer(adiv)
        return QA(full_question, full_answer)

    @staticmethod
    def _extract_question(qdiv):
        return qdiv.get_text(strip=True).replace("Q:", "").strip()

    @staticmethod
    def _extract_answer(tag: Tag) -> str:
        content = []
        for p in tag.find_all("p"):
            answer = p.get_text().replace("A:", "").strip()
            if answer:
                content.append(answer + "." if not answer.endswith(".") else answer)

        return "\n".join(content)


import yaml


def save_scrape(scraper, dest_filename):
    kb_file = Path("/Users/jlinho/Desktop/gr") / dest_filename
    with open(kb_file, "w") as f:
        items = []
        for i in scraper.qa:
            items.append(i.question)
            items.append(i.answer)
            # items.append({"question": i.question, "answer": i.answer})
        yaml.dump(items, f, sort_keys=False)


if __name__ == "__main__":
    # scrapers = {"europcar": EuropcarScraper, "fedora": FedoraScraper}
    scrapers = {"nih": NIHScraper}
    for name, scraper in scrapers.items():
        save_scrape(scraper(), f"{name}.yml")
