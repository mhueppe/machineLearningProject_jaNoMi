import json

import requests
from bs4 import BeautifulSoup
import re

# URL of the website you want to scrape
base_url = 'https://pokemondb.net'

def get_attribute(soup, key):
    """Extracts the value associated with the given key from the HTML."""
    # Find the <th> element with the specified key
    header = soup.find('th', string=key)
    if header:
        # Find the corresponding <td> element
        cell = header.find_next_sibling('td')
        if cell:
            return cell.get_text(strip=True)
    return None

def get_types(soup, key, regex = r'type-icon type-\w+'):
    """Extracts types from the HTML."""
    # Find the <th> element with the text "Type"
    type_header = soup.find('th', string=key)
    if type_header:
        # Find the corresponding <td> element
        type_cell = type_header.find_next_sibling('td')
        if type_cell:
            # Find all <a> elements within that <td> with the desired class pattern
            type_icon_elements = type_cell.find_all(class_=re.compile(regex))
            # Extract href and text from each matching element
            return [element.get_text() for element in type_icon_elements]
    return []

def getEntries(pokedex_hrefs):
    # Extract the text from each element
    # Print the extracted hrefs
    entries_all_pokemon = {}

    for i, (name, href) in enumerate(pokedex_hrefs):
        # URL of the website you want to scrape
        entries_all_pokemon[name] = {}
        pokedex_url = base_url + href
        print(f"Parsed {name}, {i}/{len(pokedex_hrefs)}")
        # Send a GET request to the website
        response = requests.get(pokedex_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all elements with the class 'cell-med-text'
            texts = soup.find_all(class_='cell-med-text')

            # Extract the text from each element
            pokedex_entries = [text.get_text(strip=True) for text in texts]

            # Print the extracted Pokédex entries
            entries_all_pokemon[name]["descriptions"] = pokedex_entries

            for a, regex in [("Type", r'type-icon type-\w+'), ("Abilities", "text-muted")]:
                entries_all_pokemon[name][a] = get_types(soup, key=a, regex=regex)

            for a in ["Height", "Weight", "Species"]:
                entries_all_pokemon[name][a] = get_attribute(soup, a)


        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
    return entries_all_pokemon

def getPokemonSites():
    all_url = base_url + '/pokedex/all'  # Replace with the actual URL
    # Send a GET request to the website
    response = requests.get(all_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all elements with the class 'ent-name'
        ent_name_elements = soup.find_all(class_='ent-name')

        # Extract the href from each element
        pokedex_hrefs = [(element.get_text(strip=True), element['href']) for element in ent_name_elements if
                         'href' in element.attrs]

        return pokedex_hrefs
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
    return []

if __name__ == '__main__':
    pokedex_hrefs = getPokemonSites()
    entries_all_pokemon = getEntries(pokedex_hrefs)
    json.dump(entries_all_pokemon, open("data/pokemon_data.json", "w"), indent=4, default=lambda x: "None")