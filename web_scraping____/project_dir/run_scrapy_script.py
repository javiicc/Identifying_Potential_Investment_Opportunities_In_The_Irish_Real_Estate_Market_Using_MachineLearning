from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from myproject.spiders.rent import RentIeSpider
from myproject.spiders.daft import DaftSpider
from myproject.spiders.property import PropertySpider

from scrapy.signalmanager import dispatcher


def spider_results():
    results = []

    def crawler_results(signal, sender, item, response, spider):
        results.append(item)

    dispatcher.connect(crawler_results, signal=signals.item_passed)

    process = CrawlerProcess(get_project_settings())
    #process.crawl(RentIeSpider)
    process.crawl(DaftSpider)
    #process.crawl(PropertySpider)
    process.start()  # the script will block here until the crawling is finished
    return results

spider_results()
#if __name__ == '__main__':
 #   print(spider_results())