import csv
import json
from datetime import date
from typing import List, Dict

type_filter = {'white', 'red', 'rose'}
reqs = {'winery', 'region', 'year', 'country'}


class Wine:
    sku: str
    created: date
    grape: str
    wine_type: str
    sparkling: bool
    region: str
    country: str
    winery: str
    name: str
    year: int
    content: float

    def __init__(self, sku: str, created: date, grape: str, wine_type: str, sparkling: bool,
                 region: str, country: str, winery: str, name: str, year: int, content: float = 0.75):
        self.content = content
        self.year = year
        self.name = name
        self.winery = winery
        self.country = country
        self.region = region
        self.sparkling = sparkling
        self.wine_type = wine_type
        self.grape = grape
        self.created = created
        self.sku = sku

    @staticmethod
    def from_row(row: dict):
        return Wine(
            row['sku'], row['created'], row['grape'], row['wine_type'], row['sparkling'] == 'True',
            row['region'], row['country'], row['winery'], row['name'], row['year']
        )

    def __hash__(self):
        return self.sku.__hash__()

    def __repr__(self):
        return f'Wine(sku={self.sku}, sparkling={self.sparkling}, ' \
               f'name={self.name}, year={self.year}, grape={self.grape}' \
               f')'


class Winery:

    def __init__(self, name: str):
        self.name = name
        self.wines = dict()

    def __iadd__(self, other):
        if isinstance(other, Wine):
            self.wines[other.sku] = other
        return self

    def __contains__(self, item):
        if isinstance(item, Wine):
            return item.winery in self.wines
        elif isinstance(item, str):
            return item in self.wines

        return False

    def __getitem__(self, item):
        if isinstance(item, Wine):
            return self.wines[item.sku]
        elif isinstance(item, str):
            return self.wines[item]
        raise KeyError

    def __hash__(self):
        return self.name.__hash__()

    def __repr__(self):
        return f'Winery(name={self.name}, wine_count={self.count_wines()}, wines={self.wines.values()})'

    def count_wines(self):
        return len(self.wines)


class Region:

    def __init__(self, name: str):
        self.name = name
        self.wineries = dict()

    def __iadd__(self, other):
        if isinstance(other, Winery):
            self.wineries[other.name] = other
        return self

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.wineries
        elif isinstance(item, Winery):
            return item in self.wineries.values()

        return False

    def __getitem__(self, item):
        if isinstance(item, Winery):
            return self.wineries[item.name]
        elif isinstance(item, str):
            return self.wineries[item]

        raise KeyError

    def __hash__(self):
        return self.name.__hash__()

    def __repr__(self):
        return f'Region(name={self.name}, wine_count={self.count_wines()}, wineries={self.wineries.values()})'

    def count_wines(self):
        total = 0
        for winery in self.wineries.values():
            total += winery.count_wines()
        return total


class Country:

    def __init__(self, name: str):
        self.name = name
        self.regions = dict()

    def __iadd__(self, other):
        if isinstance(other, Region):
            self.regions[other.name] = other
        return self

    def __contains__(self, item):
        if isinstance(item, (Region, str)):
            return item in self.regions
        return False

    def __getitem__(self, item):
        if isinstance(item, Region):
            return self.regions[item.name]
        elif isinstance(item, str):
            return self.regions[item]
        raise KeyError

    def __hash__(self):
        return self.name.__hash__()

    def __repr__(self):
        return f'Country(name={self.name}, wine_count={self.count_wines()}, regions={self.regions.values()})'

    def count_wines(self):
        total = 0
        for region in self.regions.values():
            total += region.count_wines()
        return total

    def drop_regions_below(self, count):
        self.regions = {name: region for (name, region) in self.regions.items() if region.count_wines() >= count}


def read_feed() -> List[Wine]:
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
