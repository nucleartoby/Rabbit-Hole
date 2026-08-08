import pytest
from rabbithole.core import Entry


@pytest.fixture
def entry():

    def make(title: str, summary: str = "A summary.") -> Entry:
        return Entry(
            title=title,
            summary=summary,
            url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",)

    return make
