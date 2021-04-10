import csv
from collections import defaultdict

type_filter = {'white', 'red', 'rose'}
reqs = {'winery', 'region', 'year', 'country'}

first_pass = []
with open('../feeds/embedding_feed.csv', newline='\n', encoding='utf-8') as embedding_csv:
    csv_reader = csv.DictReader(embedding_csv)
    try:
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

            first_pass.append(row)
    except UnicodeDecodeError as ude:
        line = ude.object.replace(ude.object[ude.start:ude.end], bytes("", "UTF-8")).splitlines()
        print(line)

converted_amt = len(first_pass)
second_pass = []

countries = defaultdict(int)
regions = defaultdict(int)
wineries = defaultdict(int)
region_to_country = dict()

for proper_item in first_pass:
    country = proper_item['country']
    region = proper_item['region']
    winery = proper_item['winery']

    countries[country] += 1
    regions[region] += 1
    wineries[winery] += 1
    if region in region_to_country and region_to_country[region] != country:
        print(region_to_country[region])
        print(region)
        print(country)
        print('\n')
    else:
        region_to_country[region] = country

for region in regions.copy():
    if regions[region] < 30:
        regions.pop(region)

for country in countries.copy():
    if countries[country] < 50:
        countries.pop(country)

for proper_item in first_pass:
    country = proper_item['country']
    region = proper_item['region']
    if country in countries and region in regions:
        second_pass.append(proper_item)


print(countries)
print(regions)
print(len(second_pass))
