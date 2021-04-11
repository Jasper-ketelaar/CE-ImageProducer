import csv
import json
from typing import List, Dict

from model.structure import Wine, Winery, Region, Country


def read_feed() -> List[Wine]:
    type_filter = {'white', 'red', 'rose'}
    reqs = {'winery', 'region', 'year', 'country'}
    feed_array: List[Wine] = []
    with open('../feeds/embedding_feed.csv', newline='\n', encoding='utf-8') as embedding_csv:
        csv_reader = csv.DictReader(embedding_csv)
        for row in csv_reader:
            invalid = False
            for key in reqs:
                if row[key] is None or row[key] == '':
                    invalid = True
                    break
            if invalid:
                continue

            if row['type'] != 'simple' or row['has_360'] != 'True':
                continue

            if row['wine_type'] not in type_filter or '0.75' not in row['size']:
                continue

            wine = Wine.from_row(row)
            feed_array.append(wine)

    return feed_array


def compute_hierarchy(wines: List[Wine]) -> Dict[str, Country]:
    countries = dict()
    for wine in wines:
        if wine.country not in countries:
            cntry = Country(wine.country)
            countries[wine.country] = cntry

        cntry = countries[wine.country]
        if wine.region not in cntry:
            region = Region(wine.region)
            cntry += region

        region = cntry[wine.region]
        if wine.winery not in region:
            winery = Winery(wine.winery)
            region += winery

        winery = region[wine.winery]
        winery += wine

    return countries


def filter_structure(
        countries: Dict[str, Country],
        region_min_count: int = 30,
        country_min_count: int = 50
):
    countries_updated = {name: cty for name, cty in countries.items() if cty.count_wines() >= country_min_count}
    for country in countries_updated.values():
        country.drop_regions_below(region_min_count)

    return countries_updated


if __name__ == '__main__':
    wine_list = read_feed()
    countries_hierarchical = compute_hierarchy(wine_list)
    dropped = filter_structure(countries_hierarchical)


    def default_encode(obj):
        if isinstance(obj, (Country, Region, Winery)):
            wines = obj.count_wines()
            res = vars(obj)
            res['wine_count'] = wines

            if isinstance(obj, Country):
                res['region_count'] = len(obj.regions)
            elif isinstance(obj, Region):
                res['winery_count'] = len(obj.wineries)

            return res
        elif isinstance(obj, Wine):
            return vars(obj)


    with open('../feeds/embedding_feed.json', 'w') as output_file:
        json.dump(dropped, output_file, indent=4, sort_keys=True, default=default_encode)
