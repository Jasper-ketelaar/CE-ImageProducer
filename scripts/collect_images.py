import json
import os
from datetime import date
from typing import Dict

import requests

from scripts.model.structure import Wine, Winery, Region, Country


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


def dump_wine_image(wine: Wine, image_formats=None, image_ext="png"):
    def _inner_download(formats, ext):
        for image_format in formats:
            for angle in interesting_angles:
                url = s3_format.format(wine.sku, angle, image_format, ext)
                response = requests.get(url)
                if response.status_code == 200:
                    os.makedirs(directory, exist_ok=True)
                    file_path = f'{directory}{angle}-{image_format}.{ext}'
                    if os.path.exists(file_path):
                        continue

                    with open(file_path, 'wb') as file:
                        file.write(response.content)
                else:
                    return False

    def _inner_retry():
        all_formats = [image_formats_1, image_formats_2, image_formats_3, image_formats_4]
        formats_wo_curr = all_formats.copy()
        formats_wo_curr.remove(image_formats)

        for format_entry in formats_wo_curr:
            if _inner_download(format_entry, image_ext) is not False:
                return True
        for format_entry in all_formats:
            if _inner_download(format_entry, "jpg") is not False:
                return True
        return False

    directory = f'../originals/{wine.sku}/'
    if os.path.exists(directory):
        return

    s3_format = "https://s3.eu-central-1.amazonaws.com/360.grandcruwijnen.nl/{0}/images/a_0_{1}_{2}_0_0.{3}"
    image_formats_1 = [44, 88, 132, 176]
    image_formats_2 = [37, 73, 146, 219, 293]
    image_formats_3 = [64, 128, 256, 384, 512]
    image_formats_4 = [85, 171, 341, 512, 683]

    interesting_angles = [0, 1, 5, 6, 7, 11]
    if image_formats is None:
        image_formats = image_formats_3 if wine.created < date(2020, 11, 29) else image_formats_1

    if _inner_download(image_formats, image_ext) is False:
        if _inner_retry() is False:
            print(f'No combination found for {wine.sku}')
            return

    print(f'Dumped the images for wine {wine.sku} ({wine.name})')


if __name__ == '__main__':
    structure = read_json_feed()
    for country in structure.values():
        for region in country.regions.values():
            for winery in region.wineries.values():
                for wine in winery.wines.values():
                    dump_wine_image(wine)

