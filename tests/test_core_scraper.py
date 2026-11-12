from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

import pytest

from scrapy import Request, Spider, signals
from scrapy.core import scraper as scraper_module
from scrapy.core.scraper import Scraper, _Slot
from scrapy.exceptions import IgnoreRequest, ScrapyDeprecationWarning
from scrapy.http import Response
from scrapy.utils.defer import maybe_deferred_to_future
from scrapy.utils.spider import DefaultSpider
from scrapy.utils.test import get_crawler
from tests.spiders import SimpleSpider
from tests.utils.decorators import coroutine_test, inline_callbacks_test

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator, Iterable

    from twisted.internet.defer import Deferred

    from tests.mockserver.http import MockServer


@coroutine_test
async def test_scraper_exception(
    mockserver: MockServer,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crawler = get_crawler(SimpleSpider)
    monkeypatch.setattr(
        "scrapy.core.engine.Scraper.handle_spider_output_async",
        lambda *args, **kwargs: 1 / 0,
    )
    await crawler.crawl_async(url=mockserver.url("/"))
    assert "Scraper bug processing" in caplog.text


@coroutine_test
async def test_open_spider_without_spider() -> None:
    scraper = Scraper(get_crawler(DefaultSpider))
    with pytest.raises(RuntimeError, match=r"called before Crawler\.spider is set"):
        await scraper.open_spider_async()


@inline_callbacks_test
def test_enqueue_scrape_without_slot() -> Generator[Deferred[Any], Any, None]:
    scraper = Scraper(get_crawler(DefaultSpider))
    with pytest.raises(RuntimeError, match="Scraper slot not assigned"):
        yield scraper.enqueue_scrape(Response("data:,"), Request("data:,"))


@coroutine_test
async def test_enqueue_scrape_wrong_type(caplog: pytest.LogCaptureFixture) -> None:
    crawler = get_crawler(DefaultSpider)
    crawler.spider = crawler._create_spider()
    scraper = Scraper(crawler)
    await scraper.open_spider_async()
    try:
        await maybe_deferred_to_future(
            scraper.enqueue_scrape("not a response", Request("data:,"))  # type: ignore[arg-type]
        )
    finally:
        await scraper.close_spider_async()
    assert "expected Response or Failure, got <class 'str'>" in caplog.text


@coroutine_test
async def test_call_spider_sets_request_on_response() -> None:
    class ParsingSpider(Spider):
        name = "parsing"

        def parse(self, response: Response) -> Iterable[Any]:
            return [response.url]

    crawler = get_crawler(ParsingSpider)
    crawler.spider = crawler._create_spider()
    scraper = Scraper(crawler)
    response = Response("data:,")
    request = Request("data:,")
    output = await scraper.call_spider_async(response, request)
    assert response.request is request
    assert list(cast("Iterable[Any]", output)) == ["data:,"]


@coroutine_test
async def test_errback_error_on_ignored_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An errback failing on an ignored request does not log a download error."""

    class IgnoreMiddleware:
        def process_request(self, request: Request) -> None:
            raise IgnoreRequest("nope")

    class ErrbackSpider(Spider):
        name = "errback"

        async def start(self) -> AsyncIterator[Any]:
            yield Request("data:,", errback=self.errback)

        def errback(self, failure: Any) -> None:
            raise ValueError("errback error")

    crawler = get_crawler(
        ErrbackSpider, {"DOWNLOADER_MIDDLEWARES": {IgnoreMiddleware: 1}}
    )
    with caplog.at_level(logging.DEBUG):
        await crawler.crawl_async()
    assert "Error downloading" not in caplog.text
    assert crawler.stats.get_value("spider_exceptions/ValueError") == 1


@coroutine_test
async def test_none_in_callback_output() -> None:
    """``None`` in the callback output is dropped, without dropping later items."""

    class NoneSpider(Spider):
        name = "none"

        async def start(self) -> AsyncIterator[Any]:
            yield Request("data:,")

        def parse(self, response: Response) -> Iterable[Any]:
            yield {"index": 1}
            yield None
            yield {"index": 2}

    crawler = get_crawler(
        NoneSpider,
        # The built-in spider middlewares would drop the None before the
        # scraper gets to see it. A single concurrent item makes sure that the
        # None and the item that follows it are handled by the same worker.
        {"CONCURRENT_ITEMS": 1, "SPIDER_MIDDLEWARES_BASE": {}},
    )
    await crawler.crawl_async()
    assert crawler.stats.get_value("item_scraped_count") == 2


@coroutine_test
async def test_slot_deprecated_attributes() -> None:
    values: dict[str, Any] = {}

    class SizeSpider(Spider):
        name = "size"

        async def start(self) -> AsyncIterator[Any]:
            yield Request("data:,a")

        def parse(self, response: Response) -> None:
            engine = self.crawler.engine
            assert engine is not None
            slot = engine.scraper.slot
            assert slot is not None
            values["response_active_size"] = engine.response_active_size
            for name in ("active_size", "max_active_size"):
                with pytest.warns(ScrapyDeprecationWarning, match=name):
                    values[name] = getattr(slot, name)
            with pytest.warns(ScrapyDeprecationWarning, match="needs_backout"):
                values["needs_backout"] = slot.needs_backout()

    crawler = get_crawler(SizeSpider, {"RESPONSE_MAX_ACTIVE_SIZE": 1})
    await crawler.crawl_async()
    assert values == {
        "response_active_size": 1,
        "active_size": 1,
        "max_active_size": 1,
        "needs_backout": True,
    }


def test_slot_deprecated() -> None:
    with pytest.warns(ScrapyDeprecationWarning, match="Slot is deprecated"):
        assert scraper_module.Slot is _Slot


def test_missing_attribute() -> None:
    with pytest.raises(AttributeError):
        scraper_module.Foo


@pytest.mark.only_asyncio
@coroutine_test
async def test_item_cancelled_error_does_not_stall_scraper(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A CancelledError raised while processing one item is treated as an item
    error: it is logged, the other items keep being processed and the spider
    closes normally instead of stalling with the request stuck in the slot."""

    class CancellingPipeline:
        async def process_item(self, item: Any) -> Any:
            if item["id"] == 2:
                # A subtask cancelled inside the pipeline, e.g. by a timeout,
                # whose CancelledError is not swallowed on the spot.
                task = asyncio.create_task(asyncio.Event().wait())
                task.cancel()
                await task
            return item

    class CancellingSpider(Spider):
        name = "cancelling"

        async def start(self) -> AsyncIterator[Any]:
            yield Request("data:,")

        def parse(self, response: Response) -> Iterable[Any]:
            for i in range(5):
                yield {"id": i}

    scraped: list[Any] = []

    def on_item_scraped(item: Any, **kwargs: Any) -> None:
        scraped.append(item)

    crawler = get_crawler(
        CancellingSpider,
        {"ITEM_PIPELINES": {CancellingPipeline: 1}},
    )
    crawler.signals.connect(on_item_scraped, signals.item_scraped)
    # Without the fix the crawl never finishes, so this turns the hang into a
    # test failure instead of blocking the whole suite.
    async with asyncio.timeout(10):
        await crawler.crawl_async()
    assert sorted(item["id"] for item in scraped) == [0, 1, 3, 4]
    assert crawler.stats.get_value("item_scraped_count") == 4
    assert "Error processing an item" in caplog.text
    assert "CancelledError" in caplog.text
