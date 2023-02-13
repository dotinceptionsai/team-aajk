import abc
import re
import time
import typing
from dataclasses import dataclass
from typing import Iterable

import requests
from bs4 import BeautifulSoup, Tag

SCRAPER_TYPE = typing.TypeVar("SCRAPER", bound="Scraper")


def sentence_ended(sentence: str) -> bool:
    return sentence.endswith(".") or sentence.endswith("?") or sentence.endswith("!")


@dataclass
class QA:
    question: str
    answer: str

    def __init__(self, question: str, answer: str):
        self.question = question.replace("\u00A0", " ").strip()
        self.answer = answer.replace("\u00A0", " ").strip()


class Scraper(abc.ABC):
    name: str
    _registered_types: dict[str, typing.Type[SCRAPER_TYPE]] = {}

    @abc.abstractmethod
    def scrape(self) -> Iterable[QA]:
        ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registered_types[cls.name] = cls

    @classmethod
    def all(cls) -> dict[str, "Scraper"]:
        return {
            scraper_name: scraper_type()
            for scraper_name, scraper_type in cls._registered_types.items()
        }


class NIHScraper(Scraper):
    name = "nih"
    url = "https://search.grants.nih.gov/faq/api/faq/664"

    def scrape(self) -> Iterable[QA]:
        qas: list[QA] = []

        resp = requests.get(self.url, headers={"Accept": "application/json"})
        body = resp.json()

        for data in body["data"]:
            for header in data["headers"]:
                for json_qa in header["questions"]:
                    qas.append(self.extract_qa(json_qa))
        return qas

    @staticmethod
    def extract_qa(json_qa) -> QA:
        question = json_qa["Question"]
        a = BeautifulSoup(json_qa["Answer"].replace("&nbsp;", ""), "html.parser")
        answer_paragraphs = [p.get_text(strip=False) for p in a.find_all("p")]
        if (
            len(answer_paragraphs) == 0
        ):  # Some answers (4 out of 125) do not start with the <p> tag
            answer_paragraphs = [a.get_text(strip=False)]

        answer = ""
        for p in answer_paragraphs:
            p = p.strip()
            suffix = " " if sentence_ended(p) else ". "
            answer += p + suffix
        return QA(question, answer.strip())


class OlympicsScraper(Scraper):
    name = "olympics"
    site = "https://olympics.com"
    root_page = "/ioc/faq"

    def scrape(self) -> Iterable[QA]:
        qas: list[QA] = []

        root_page = requests.get(
            self.site + self.root_page, headers={"User-Agent": "Mozilla/5.0"}
        )
        root_page_soup = BeautifulSoup(root_page.content, "html.parser")

        for child_page in self.child_pages(root_page_soup):
            time.sleep(0.5)  # Pause avoiding to be rejected by website
            qas.extend(self._child_page_qas(self.site + child_page))
        return qas

    @staticmethod
    def child_pages(root_page_soup: BeautifulSoup):
        for a in root_page_soup.find_all("a", {"class": "btn-ioc"}):
            if a.get_text(strip=True).startswith("View All Questions"):
                yield a["href"]

    @staticmethod
    def _child_page_qas(child_page_url: str) -> Iterable[QA]:
        print(f"Visiting {child_page_url} ...")
        qas: list[QA] = []

        page = requests.get(child_page_url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(page.content, "html.parser")

        for q_item in OlympicsScraper._find_qa_sections(soup):
            qas.append(OlympicsScraper._extract_qa(q_item))

        return qas

    @staticmethod
    def _find_qa_sections(soup: BeautifulSoup) -> Iterable[BeautifulSoup]:
        for ul in soup.find_all("li"):
            if "data-accordion-item" in ul.attrs:
                yield ul

    @staticmethod
    def _extract_qa(item: Tag) -> QA:
        q_candidate = [
            d for d in item.find_all("div") if "data-accordion-opener-title" in d.attrs
        ]
        assert len(q_candidate) == 1, "Could not find question div"
        question = q_candidate[0].get_text(strip=True)

        q_answer = [
            d for d in item.find_all("ul") if "data-accordion-content" in d.attrs
        ]
        assert len(q_answer) == 1, "Could not find answer ul"
        answer = (
            q_answer[0]
            .get_text(strip=True, separator=" ")
            .replace("\n", " ")
            .replace("\u201C", '"')
            .replace("\u201D", '"')
        )
        answer = re.sub(r"\s+", " ", answer)
        if (idx := answer.lower().find("learn more:")) >= 0:
            answer = answer[:idx].strip()

        return QA(question, answer)


class EuropcarScraper(Scraper):
    name = "europcar"
    root_url = "https://faq.europcar.com/"
    max_depth = 3

    def scrape(self) -> Iterable[QA]:
        qas: list[QA] = []
        self._feed_qa_items(qas, self.root_url, self.max_depth)
        return qas

    @staticmethod
    def _feed_qa_items(qas: list[QA], url: str, remaining_depth: int):
        print(f"Visiting {url} ...")
        if remaining_depth == 0:
            return

        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")

        for child_uri in EuropcarScraper._find_qa_page_links(soup):
            time.sleep(0.5)  # Pause avoiding to be rejected by website
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


class FedoraScraper(Scraper):
    name = "fedora"
    url = "https://fedoraproject.org/wiki/FAQ#Getting_Started"

    def scrape(self) -> Iterable[QA]:
        qas: list[QA] = []
        page = requests.get(self.url)
        soup = BeautifulSoup(page.content, "html.parser")

        for q_item in self._find_qa_sections(soup):
            qas.append(self._extract_qa(q_item))
        return qas

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
        while (next_p := next_p.find_next_sibling()) and FedoraScraper._is_para(next_p):
            answer = next_p.get_text().strip()
            if answer:
                content.append(answer)
        full_answer = "\n".join(content)
        return full_answer

    @staticmethod
    def _is_para(tag: Tag) -> bool:
        return tag is not None and tag.name == "p"


class WwfScraper(Scraper):
    name = "wwf"
    url = "https://www.wwf.org.uk/faqs"

    def scrape(self) -> Iterable[QA]:
        qas: list[QA] = []
        page = requests.get(self.url)
        soup = BeautifulSoup(page.content, "html.parser")

        for q_item in WwfScraper._find_qa_sections(soup):
            qas.append(WwfScraper._extract_qa(q_item))
        return qas

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


class FdaCovidScraper(Scraper):
    name = "fda"
    site = "https://www.fda.gov/"
    page = "emergency-preparedness-and-response/coronavirus-disease-2019-covid-19/covid-19-frequently-asked-questions"

    def scrape(self) -> Iterable[QA]:
        qas: list[QA] = []
        url = self.site + self.page
        print(f"Visiting {url} ...")

        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")

        for q_item in self._find_qa_sections(soup):
            qas.append(self._extract_qa(q_item))
        return qas

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
