import concurrent
import json
import os
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Dict, List

import requests

from images.model.structure import Wine, Winery, Region, Country

s3_format: str = "https://s3.eu-central-1.amazonaws.com/360.grandcruwijnen.nl/{0}/images/a_0_{1}_{2}_0_0.{3}"


def read_json_feed() -> Dict[str, Country]:
    def load_hierarchy(dct: dict):
        countries: Dict[str, Country] = dict()
        for country in dct.values():
            country_mapped = Country(country['name'])

            for region in country['regions'].values():
                region_mapped = Region(region['name'])

                for winery in region['wineries'].values():
                    winery_mapped = Winery(winery['name'])

                    for wine in winery['wines'].values():
                        wine_mapped = Wine.from_row(wine)
                        winery_mapped += wine_mapped
                    region_mapped += winery_mapped
                country_mapped += region_mapped
            countries[country_mapped.name] = country_mapped

        return countries

    with open('../feeds/embedding_feed.json') as feed:
        return load_hierarchy(json.load(feed))


class WineImageDumper:

    def __init__(self, wine: Wine):
        self.wine = wine
        self.format_types = [
            [44, 88, 132, 176],
            [37, 73, 146, 219, 293],
            [64, 128, 256, 384, 512],
            [85, 171, 341, 512, 683],
        ]
        self.interesting_angles = [0, 1, 5, 6, 7, 11]
        self.responses = dict()

    def perform_async(self, ext='png'):
        urls_ft = []
        format_types_len = len(self.format_types)
        for fi in range(format_types_len):
            urls_ft.append([])
            for res in self.format_types[fi]:
                for angle in self.interesting_angles:
                    urls_ft[fi].append(s3_format.format(self.wine.sku, angle, res, ext))

        form = 0
        for urls in urls_ft:
            form = len(urls)
            with ThreadPoolExecutor(max_workers=form) as executor:
                futures = {executor.submit(requests.get, url): url for url in urls}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result: requests.Response = future.result()
                        res_end_index = result.url.index(f'_0_0.{ext}')
                        res_begin_index = result.url.index(f'a_0_') + 4
                        angle_res = result.url[res_begin_index:res_end_index].replace('_', '-')

                        if result.status_code != 200:
                            break

                        self.responses[f'{angle_res}.{ext}'] = result.content
                    except TimeoutError:
                        print("Timeout Error")
            if len(self.responses) > 0:
                break

        response_length = len(self.responses)
        return form >= response_length > 0


def dump_wine_image(wine: Wine) -> List[str]:
    dumper = WineImageDumper(wine)
    directory = f'../originals/{wine.sku}/'
    if os.path.exists(directory):
        files = os.listdir(directory)
        if len(files) >= 24 and any('44' in file for file in files):
            return []
        elif len(files) >= 30:
            return []

    failed = []
    if dumper.perform_async() is True:
        print(f'Dumped {len(dumper.responses)} images for wine {wine.sku} ({wine.name})')
        for name in dumper.responses:
            res = dumper.responses[name]
            with open(f'{directory}{name}', 'wb') as save:
                save.write(res)
                save.close()
    elif dumper.perform_async('jpg') is False:
        print(f"Failed to dump images for {wine.sku} even in jpg")
        failed.append(wine.sku)

    return failed


def dump_all_images():
    structure = read_json_feed()
    skus_mistake = set()

    for country in structure.values():
        for region in country.regions.values():
            for winery in region.wineries.values():
                for wine in winery.wines.values():
                    skus_mistake.update(dump_wine_image(wine))

    return skus_mistake


if __name__ == '__main__':
    print(dump_all_images())
